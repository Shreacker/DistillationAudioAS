import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from importlib import import_module

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

@torch.no_grad()
def divergence_metric(metric, P, Q, dev_loader, device='cpu'):
    p_mode, q_mode = P.training, Q.training
    P.eval()
    Q.eval()

    total_div, n = 0., 0
    for item in dev_loader:
        # Check key name
        X = item['data'].to(device).flatten(1)
        log_p = F.log_softmax(P(X), dim=1)
        log_q = F.log_softmax(Q(X), dim=1)
        div = metric(log_p, log_q)
        total_div += div.sum().item()
        n += X.size(0)

    P.train(p_mode)
    Q.train(q_mode)
    return total_div / n

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

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
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
                            pin_memory=True)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader

def _get_optimizer(model_parameters, optim_config):
    optimizer_name = optim_config['optimizer']

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
                        lr=optim_config['base_lr'],
                        momentum=0.9,
                        betas=optim_config['betas'],
                        weight_decay=optim_config['weight_decay'],
                        amsgrad=str_to_bool(optim_config['amsgrad'])
                        )
    else:
        print('Un-known optimizer', optimizer_name)
        sys.exit()

    return optimizer

def _get_scheduler(optimizer, optim_config):
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

def GS(
        student_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        teacher_checkpoints: list,
        current_idx: int=-1,
        dev_dl: DataLoader=None,
        threshold: float=0.8,
        metric=None,
        eps: float=1e-12,
        device='cpu'
):
    student_model.eval()

    j = current_idx + 1
    n_anchors = len(teacher_checkpoints)

    def anchor2metric(idx):
        if idx >= 0:
            anchor = torch.load(teacher_checkpoints[idx], map_location=device, weights_only=True)
            teacher_model.load_state_dict(anchor)
        
        return divergence_metric(metric, student_model, teacher_model, dev_dl, device=device)

    h_i = anchor2metric(current_idx)

    while j < n_anchors:
        h_j = anchor2metric(j)
        r_ij = (h_j - h_i) / h_i
        if r_ij > threshold:
            return j - 1

    return n_anchors - 1