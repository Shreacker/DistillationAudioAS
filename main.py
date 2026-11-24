import os, sys
os.makedirs('./models/checkpoints/students', exist_ok=True)

import argparse
import json
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from shutil import copy
from natsort import natsorted

from utils.loss_fn import ClassroomCache, ClassroomLoss
from utils.ga import EnsembleWeightGA
from utils.evaluation import calculate_tDCF_EER
from utils.utilities import str_to_bool, set_seed, get_model, get_loader, create_optimizer, GS, collect_log_probs, cache_anchor_log_probs, anchor_cache_path, load_anchor_cache

warnings.filterwarnings('ignore', category=FutureWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args: argparse.Namespace) -> None:

    # Load students' configurations
    with open(args.config, 'r') as f:
        config = json.loads(f.read())

    with open(args.teacher_config, 'r') as f:
        teacher_config = json.loads(f.read())

    model_config = {}
    optim_config = {}
    track = config['track']
    assert track in ['LA', 'PA', 'DF'], 'Invalid track given'
    for i in range(5):
        s = f'student{i+1}'
        model_config[s] = config[s]['model_config']
        optim_config[s] = config[s]['optim_config']
        optim_config[s]['epochs'] = config['num_epochs']

    seed = args.seed
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    database_path = Path(config['database_path'])
    teacher_chkps_path = Path(config['teacher']['checkpoints_path'])
    prefix_la_2019 = f'ASVspoof2019.LA'
    dev_trial_path = (database_path /
                      f"ASVspoof2019_LA_cm_protocols/{prefix_la_2019}.cm.dev.trl.txt")
    eval_trial_path = (database_path /
                       f"ASVspoof2019_LA_cm_protocols/{prefix_la_2019}.cm.eval.trl.txt")

    # Model related paths
    model_tag = '{}_{}_ep{}_bs{}'.format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config['num_epochs'], config['batch_size']
    )
    if args.comment:
        model_tag = model_tag + f"_{args.comment}"
    model_tag = output_dir / model_tag
    model_save_path = model_tag / 'weights'
    ens_weight_save_path = model_tag / 'ens_weights'
    eval_score_path = model_tag / config['eval_output']
    writer = SummaryWriter(model_tag)
    for path in [model_tag, model_save_path, ens_weight_save_path, eval_score_path]:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {path}: {e}")
    copy(args.config, model_tag / 'config.conf')

    teacher_chkps = []
    for file in natsorted(teacher_chkps_path.iterdir()):
        if file.is_file():
            teacher_chkps.append(str(file))
    assert len(teacher_chkps) > 0, "No teacher checkpoints found."

    print(f'Device: {device}')
    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')

    # Models and DataLoaders
    print('--Teacher params count--')
    teacher = get_model(teacher_config['model_config'], 'Teacher', device)
    print('--Students params count--')
    students = [get_model(model_config[s], 'Student', device) for s in model_config.keys()]
    train_dl, dev_dl, eval_dl = get_loader(
        database_path, args.seed, config
    )

    # Evaluates pretrained checkpoints
    if args.eval:
        if args.ens_weight_path:
            ens_weight = torch.load(args.ens_weight_path)
        else:
            ens_weight = torch.full((len(model_config),), 1 / len(model_config))

        for i, s in enumerate(students):
            s.load_state_dict(
                torch.load(config[list(model_config.keys())[i]]['model_path'], map_location=device)
            )
            print(f'Model Student-{i+1} loaded : {config[list(model_config.keys())[i]]['model_path']}')
        print('Start evaluation...')
        produce_evaluation_file(eval_dl, students, device,
                                ens_weight, eval_score_path, eval_trial_path)
        calculate_tDCF_EER(cm_scores_file=eval_score_path,
                           asv_score_file=database_path / config['asv_eval_score_path'],
                           output_file=model_tag / 't-DCF_EER.txt')
        print('DONE.')
        sys.exit(0)

    # Optim and Scheduler
    optimizers, schedulers = [], []
    for s, m in zip(model_config.keys(), students):
        optim_config[s]['steps_per_epoch'] = len(train_dl)
        o, sc = create_optimizer(list(m.parameters()), optim_config[s])
        optimizers.append(o)
        schedulers.append(sc)

    best_dev_eer = 100.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    f_log = open(model_tag / 'metric_log.txt', 'a')
    f_log.write('=' * 5 + '\n')

    metric_path = model_tag / 'metrics'
    os.makedirs(metric_path, exist_ok=True)

    ###
    val_idx = torch.randperm(len(dev_dl.dataset), generator=torch.Generator().manual_seed(args.seed))[:1000].tolist()
    val_dl = DataLoader(Subset(dev_dl.dataset, val_idx), batch_size=dev_dl.batch_size, shuffle=False)
    current_index = -1
    threshold = config['teacher'].get('threshold', 0.8)
    ###
    anchor_cache = args.teacher_anchor_cache or anchor_cache_path(seed, len(val_idx), teacher_chkps)
    if os.path.isfile(anchor_cache):
        anchor_log_p = load_anchor_cache(anchor_cache, val_idx, teacher_chkps)
        tqdm.write(f'Loaded teacher anchor cache: {anchor_cache}')
    else:
        anchor_log_p = cache_anchor_log_probs(teacher, teacher_chkps, val_dl, anchor_cache, seed, val_idx, device)
        tqdm.write(f'Saved teacher anchor cache: {anchor_cache}')

    criterion = ClassroomLoss(weight=torch.tensor([0.1, 0.9])).to(device)
    cache = ClassroomCache(len(students), len(train_dl.dataset), n_classes=2)
    labels_all = torch.zeros(len(train_dl.dataset), dtype=torch.int64)
    ens_weight = torch.full((len(model_config),), 1 / len(model_config))
    ens_weight_hist = []
    n_phase = len(optim_config['student1']['alpha'])

    # Training
    for epoch in tqdm(range(config['num_epochs']), desc='Epoch', leave=True, position=0):
        tqdm.write(f'Start training epoch {epoch:03d}')
        if current_index < len(teacher_chkps) - 1:
            ###
            new_index = min(
                GS(collect_log_probs(s, val_dl, device, desc=f'Collect student {k+1}\'s cache'), anchor_log_p, current_index, threshold, desc=f'GS Student {k+1}: ')
                for k, s in enumerate(students)
            )
            if new_index != current_index:
                current_index = new_index
                teacher.load_state_dict(torch.load(teacher_chkps[current_index], map_location=device))
                teacher.eval()
        
        tqdm.write(f'Teacher checkpoint index: {current_index}')
        phase = min(n_phase - 1, current_index * n_phase // len(teacher_chkps))
        epoch_losses = []
        
        for i, (s, m) in enumerate(tqdm(zip(model_config.keys(), students), total=len(students), desc='Students', leave=False, position=1)):
            oc = optim_config[s]
            hp = dict(
                    T=oc['T'],
                    alpha=oc['alpha'][phase],
                    beta=oc['beta'][phase],
                    lambda_=oc['lambda_'][phase],
                    gamma=oc['gamma']
                )
            loss_s = train_epoch(
                train_dl,
                m, i,
                teacher,
                cache,
                criterion,
                optimizers[i],
                device,
                schedulers[i],
                hp,
                ens_weight,
                labels_all,
                oc.get('scheduler')
            )
            epoch_losses.append(loss_s) # (5,)

        ga = EnsembleWeightGA(n_models=len(students))
        ###
        seen = cache.probs.sum(-1).gt(0).all(0)
        ens_weight = ga.evolve(cache.probs[:, seen].numpy(), labels_all[seen].numpy(), generations=50, patience=8)
        ens_weight_hist.append(ens_weight.copy())
        ens_loss = np.mean(epoch_losses)
        produce_evaluation_file(dev_dl, students, device, ens_weight,
                                metric_path / 'dev_score.txt', dev_trial_path)
        dev_eer, dev_tdcf = calculate_tDCF_EER(
            cm_scores_file=metric_path / 'dev_score.txt',
            asv_score_file=database_path / config['asv_dev_score_path'],
            output_file=metric_path / f'dev_t-DCF_EER_{epoch:03d}epo.txt',
            printout=False
        )
        torch.save(ens_weight_hist, ens_weight_save_path / f'ensemble-weight-log_{epoch}epo.pt')
        tqdm.write(f'ens_Loss: {ens_loss:.4f} | dev_eer: {dev_eer:3f} | dev_tdcf: {dev_tdcf:.5f} | ens_weights: {np.round(ens_weight, 2)}')
        writer.add_scalars('loss/students', {s: l for s, l in zip(model_config.keys(), epoch_losses)}, epoch)
        writer.add_scalar('loss/mean', ens_loss, epoch)
        writer.add_scalar('dev_eer', dev_eer, epoch)
        writer.add_scalar('dev_tdcf', dev_tdcf, epoch)

        # Callback to exit early
        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            tqdm.write(f'Best model found at epoch {epoch}')
            best_dev_eer = dev_eer
            for i, s in enumerate(students):
                torch.save(s.state_dict(),
                           model_save_path / f'student{i+1}_epoch_{epoch}_{dev_eer:03.3f}.pth')
            best_ens_weight = ens_weight
            torch.save(best_ens_weight, ens_weight_save_path / 'best_ens_weight.pt')

            if str_to_bool(config['eval_all_best']):
                produce_evaluation_file(eval_dl, students, device,
                                        ens_weight, eval_score_path, eval_trial_path)
                eval_eer, eval_tdcf = calculate_tDCF_EER(
                    cm_scores_file=eval_score_path,
                    asv_score_file=database_path / config['asv_eval_score_path'],
                    output_file=metric_path / f't-DCF_EER_{epoch:03d}epo.txt'
                )

                log_text = f'epoch{epoch:03d}, '
                if eval_eer < best_eval_eer:
                    log_text += f'best eer, {eval_eer:.4f}%'
                    best_eval_eer = eval_eer
                if eval_tdcf < best_eval_tdcf:
                    log_text += f'best tdcf, {eval_tdcf:.4f}'
                    best_eval_tdcf = eval_tdcf
                    for i, s in enumerate(students):
                        torch.save(s.state_dict(),
                                   model_save_path / f'student{i+1}_best.pth')
                if len(log_text) > 0:
                    tqdm.write(log_text)
                    f_log.write(log_text + '\n')

        writer.add_scalar('best_dev_eer', best_dev_eer, epoch)
        writer.add_scalar('best_dev_tdcf', best_dev_tdcf, epoch)

    print('Start final evaluation')
    epoch += 1
    produce_evaluation_file(eval_dl, students, device,
                            best_ens_weight, eval_score_path, eval_trial_path)
    eval_eer, eval_tdcf = calculate_tDCF_EER(
        cm_scores_file=eval_score_path,
        asv_score_file=database_path / config['asv_eval_score_path'],
        output_file=model_tag / 't-DCF_EER.txt'
    )
    f_log.write('=' * 5 + '\n')
    f_log.write(f'EER: {eval_eer:.3f}, min t-DCF: {eval_tdcf:.5f}')
    f_log.close()

    print(f'Training FINISHED. EER: {eval_eer:.3f}, min t-DCF: {eval_tdcf:.5f}')

def train_epoch(
        train_dl: DataLoader,
        model: nn.Module,
        s: int,
        teacher: nn.Module,
        cache: ClassroomCache,
        criterion: nn.Module,
        optim: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler,
        hp: dict,
        ens_weight,
        labels_all: torch.Tensor,
        sched_name: str | None,
):
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()
    teacher.eval()

    pbar = tqdm(train_dl, desc=f'Student {s+1}', leave=False, position=2)
    for batch_x, batch_y, idx in pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        # Teacher forward runs once per student per epoch -> Cache teacher logits
        with torch.no_grad():
            _, teacher_logits = teacher(batch_x)

        # First forward-backward pass
        optim.zero_grad()
        logits = model(batch_x)
        classroom, peers = cache.get(s, idx, device, ens_weight)
        batch_loss = criterion(logits, batch_y, teacher_logits, classroom, peers, **hp)
        running_loss += batch_loss.item() * batch_size
        pbar.set_postfix(loss=f'{running_loss / num_total:.4f}')
        batch_loss.backward()
        optim.first_step()

        # Second forward-backward pass
        optim.zero_grad()
        criterion(model(batch_x), batch_y, teacher_logits, classroom, peers, **hp).backward()
        optim.second_step()

        cache.update(s, idx, logits)
        labels_all[idx.cpu()] = batch_y.cpu()

        if scheduler is None:
            pass
        elif sched_name in ['cosine', 'keras_decay']:
            scheduler.step()
        else:
            raise ValueError(f'Scheduler error, got:{scheduler}')

    running_loss /= num_total
    cache.mark_filled(s)

    return running_loss

def produce_evaluation_file(
        dl: DataLoader,
        models: list[nn.Module],
        device: torch.device,
        ens_weight: list,
        save_path: str | list,
        trial_path: str
) -> None:
    for m in models:
        m.eval()

    with open(trial_path, 'r') as f_trl:
        trial_lines = f_trl.readlines()

    fname_list = []
    score_list = []
    for batch_x, utt_id in dl:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_scores = None
            for m in models:
                batch_out = m(batch_x).unsqueeze(dim=1) # (B, 1, 2)

                if batch_scores is None:
                    batch_scores = batch_out
                else:
                    batch_scores = torch.cat([batch_scores, batch_out], dim=1) # (B, M, 2)

            ens_out = torch.einsum('ijk,j->ik', batch_scores, torch.as_tensor(ens_weight, dtype=batch_scores.dtype, device=batch_scores.device)).cpu() # (B, 2)
        fname_list.extend(utt_id)
        score_list.extend(ens_out[:, 1].tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, 'w') as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write(f'{utt_id} {src} {key} {sco}\n')
    tqdm.write(f'Scores saved to {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial on Distillation Frameworks')
    parser.add_argument('--config',
                        dest='config',
                        type=str,
                        help='configuration file',
                        required=True
    )
    parser.add_argument('--teacher_config',
                        dest='teacher_config',
                        type=str,
                        help='teacher (AASIST) configuration file',
                        default='./config/AASIST.conf'
    )
    parser.add_argument('--teacher_anchor_cache',
                        dest='teacher_anchor_cache',
                        type=str,
                        help='anchor cache of teacher on a subset of developing dataloader',
                        default=None
    )
    parser.add_argument('--output_dir',
                        dest='output_dir',
                        type=str,
                        help='output directory for results',
                        default='./results'
    )
    parser.add_argument('--seed',
                        type=int,
                        default=21,
                        help='random seed (default: 21)'
    )
    parser.add_argument('--eval',
                        action='store_true',
                        help='when this flag is given, evaluates given model and exit'
    )
    parser.add_argument('--comment',
                        type=str,
                        default=None,
                        help='comment to describe the saved model'
    )
    parser.add_argument('--ens_weight_path',
                        type=str,
                        default=None,
                        help='directory to the model weight file (can be also given in the config file)'
    )
    main(parser.parse_args())