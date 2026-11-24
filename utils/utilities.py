import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from importlib import import_module
from tqdm import tqdm

from data_utils.ASVspoof2019 import Dataset_ASVspoof2019_train, Dataset_ASVspoof2019_devNeval, genSpoof_list
from utils.sam import SAM

def str_to_bool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError(f'Invalid truth value {val}.')

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min)*0.5*(1 + np.cos(step / total_steps * np.pi))

def keras_decay(step, decay=0.0001):
    return 1. / (1. + decay*step)

def kl_div(log_p, log_q, dim=1):
    return (log_p.exp() * (log_p - log_q)).sum(dim=dim)

def js_div(log_p, log_q):
    log_m = torch.logsumexp(torch.stack([log_p, log_q]), dim=0) - torch.log(torch.tensor(2.0))
    
    return 0.5 * kl_div(log_p, log_m) + 0.5 * kl_div(log_q, log_m)

###
@torch.no_grad()
def collect_log_probs(model, dl, device='cpu', desc=None):
    mode = model.training
    model.eval()
    out = []
    for item in tqdm(dl, desc=desc, leave=False, disable=desc is None):
        o = model(item[0].to(device))
        o = o[1] if isinstance(o, tuple) else o
        out.append(F.log_softmax(o, dim=-1).cpu())
    model.train(mode)

    return torch.cat(out)

###
def anchor_cache_path(seed, n_val, teacher_checkpoints, root='./models/checkpoints/teacher_anchor_cache'):
    return os.path.join(root, f'seed{seed}_n{n_val}_A{len(teacher_checkpoints)}.pth')

###
def cache_anchor_log_probs(teacher_model, teacher_checkpoints, dl, save_path, seed, val_idx, device='cpu'):
    res = []
    for k, c in enumerate(tqdm(teacher_checkpoints, desc='Caching teacher anchors', leave=False)):
        teacher_model.load_state_dict(torch.load(c, map_location=device, weights_only=True))
        res.append(collect_log_probs(teacher_model, dl, device, desc=f'Anchor C_{k}'))
    log_p = torch.stack(res)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save({
        'log_p': log_p,
        'seed': seed,
        'val_idx': list(val_idx),
        'chkps': [os.path.basename(c) for c in teacher_checkpoints]
    }, save_path)

    return log_p

###
def load_anchor_cache(path, val_idx, teacher_checkpoints):
    d = torch.load(path, map_location='cpu')
    assert d['val_idx'] == list(val_idx), f'Anchor cache {path} was built on a different validation subset (seed {d["seed"]}).'
    assert d['chkps'] == [os.path.basename(c) for c in teacher_checkpoints], f'Anchor cache {path} was built on different teacher checkpoints.'

    return d['log_p']

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_model(model_config: dict, model: str='Student', device: torch.device=torch.device('cpu')):
    if model.lower() == 'student':
        module = import_module(f"models.{model_config['architecture']}")
        _model = getattr(module, "Student")
    elif model.lower() == 'teacher':
        module = import_module(f"models.AASIST.models.AASIST")
        _model = getattr(module, "Model")

    m = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in m.parameters()])
    print(f"No. model params: {nb_params}")

    return m

def get_loader(
        database_path: str,
        seed: int,
        config: dict
) -> list[torch.utils.data.DataLoader]:
    track = config['track']
    prefix_2019 = f'ASVspoof2019.{track}'

    trn_database_path = database_path / f'ASVspoof2019_{track}_train/'
    dev_database_path = database_path / f'ASVspoof2019_{track}_dev/'
    eval_database_path = database_path / f'ASVspoof2019_{track}_eval/'

    trn_list_path = (database_path /
                     "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                         track, prefix_2019))
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train, labels=d_label_trn, base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)

    trn_loader = DataLoader(train_set,
                            batch_size=config['batch_size'],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available(),
                            worker_init_fn=seed_worker,
                            generator=gen)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=torch.cuda.is_available())

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=torch.cuda.is_available())

    return trn_loader, dev_loader, eval_loader

def _get_optimizer(model_parameters, optim_config):
    optimizer_name = optim_config['optimizer'].strip().lower()

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters,
                                    lr=optim_config['base_lr'],
                                    momentum=optim_config['momentum'],
                                    weight_decay=optim_config['weight_decay'],
                                    nesterov=optim_config['nesterov']
                                    )

    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model_parameters,
                                      lr=optim_config['base_lr'],
                                      betas=optim_config['betas'],
                                      weight_decay=optim_config['weight_decay'],
                                      amsgrad=str_to_bool(optim_config['amsgrad'])
                                      )
    elif optimizer_name == 'adamw+sam':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model_parameters,
                        base_optimizer=base_optimizer,
                        lr=optim_config.get('base_lr', 1e-4),
                        rho=optim_config.get('rho', 0.05),
                        betas=optim_config.get('betas', [0.9, 0.999]),
                        weight_decay=optim_config.get('weight_decay', 1e-4),
                        amsgrad=str_to_bool(optim_config.get('amsgrad', 'False'))
                        )
    else:
        print('Un-known optimizer', optimizer_name)
        sys.exit()

    return optimizer

def _get_scheduler(optimizer, optim_config):
    optimizer = getattr(optimizer, 'base_optimizer', optimizer)
    if optim_config['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config['milestones'],
            gamma=optim_config['lr_decay']
        )

    elif optim_config['scheduler'] == 'sgdr':
        pass

    elif optim_config['scheduler'] == 'cosine':
        total_steps = optim_config['epochs'] * optim_config['steps_per_epoch']
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,
                optim_config['lr_min'] / optim_config['base_lr']
            )
        )

    elif optim_config['scheduler'] == 'keras_decay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: keras_decay(step)
        )
    
    else:
        scheduler = None
        
    return scheduler

def create_optimizer(model_parameters, optim_config):
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    
    return optimizer, scheduler

###
def GS(
        student_log_p: torch.Tensor,
        anchor_log_p: torch.Tensor,
        current_idx: int=-1,
        threshold: float=0.8,
        eps: float=1e-12,
        desc: str=None,
):
    kl = (anchor_log_p.exp() * (anchor_log_p - student_log_p.unsqueeze(0))).sum(-1).mean(-1)
    n_anchors = kl.size(0)

    current_idx = max(current_idx, 0)
    h_i = kl[current_idx].clamp_min(eps)

    for j in range(current_idx + 1, n_anchors):
        r_ij = (kl[j] - h_i) / h_i

        if r_ij > threshold:
            tqdm.write(desc + f'At j={j}, r_ij={float(r_ij):.3f} > thr={threshold}')
            return j - 1

    return n_anchors - 1