import os
os.makedirs('./models/checkpoints/students', exist_ok=True)

import argparse
import json
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
from utils.sam import SAM
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from pathlib import Path
from typing import Union
from natsort import natsorted

from data_utils.mnist_dataset import MnistDataset, collate_fn
from utils.loss_fn import ClassroomCache, ClassroomLoss
from utils.ga import EnsembleWeightGA
from utils.utilities import set_seed, get_model, get_loader, create_optimizer, GS
from models.student import Student

warnings.filterwarnings('ignore', category=FutureWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args: argparse.Namespace) -> None:

    # Load students' configurations
    with open(args.config, 'r') as f:
        config = json.loads(f.read())

    # Temp - I will copy all teacher config to Student.conf later
    with open(args.teacher_config, 'r') as f:
        teacher_config = json.loads(f.read())

    model_config = {}
    optim_config = {}
    track = {}
    for i in range(len(config) - 1):
        s = f'student{i+1}'
        model_config[s] = config[s]['model_config']
        optim_config[s] = config[s]['optim_config']
        optim_config[s]['epochs'] = config[s]['num_epochs']
        track[s] = config[s]['track']
        assert track[s] in ['LA-TTS', 'LA-VC', 'LA-Both'], "Invalid track given to Student model."

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    database_path = Path(config['student1']['database_path'])
    teacher_chkps_path = Path(config['teacher']['checkpoints_path'])

    teacher_chkps = []
    for file in natsorted(teacher_chkps_path.iterdir()):
        if file.is_file():
            teacher_chkps.append(str(file))
    assert len(teacher_chkps) > 0, "No teacher checkpoints found."

    print(f'Device: {device}')

    # Models and DataLoaders
    teacher = get_model(teacher_config['model_config'], 'Teacher', device)
    students = [get_model(model_config[s], 'Student', device) for s in model_config.keys()]
    train_dl, dev_dl, eval_dl = get_loader(
        database_path, args.seed, config['student1']
    )

    # Optim and Scheduler
    optimizers, schedulers = [], []
    for s, m in zip(model_config.keys(), students):
        optim_config[s]['steps_per_epoch'] = len(train_dl)
        o, sc = create_optimizer(list(m.parameters()), optim_config[s])
        optimizers.append(o)
        schedulers.append(sc)

    best_dev_eer = 1.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.

    val_dl = DataLoader(Subset(train_dl.dataset, range(1000)), batch_size=train_dl.batch_size, shuffle=False)
    current_index = -1
    threshold = config['teacher'].get('threshold', 1.0)
    divergence_metric = nn.KLDivLoss(reduction='batchmean')

    criterion = ClassroomLoss(weight=torch.tensor([0.1, 0.9])).to(device)
    cache = ClassroomCache(len(students), len(train_dl.dataset), n_classes=2)
    labels_all = torch.zeros(len(train_dl.dataset), dtype=torch.int64)
    ens_weight = None
    n_phase = len(optim_config['student1']['alpha'])

    for epoch in range(config['student1']['num_epochs']):
        print(f'Start training epoch {epoch:03d}')
        if current_index < len(teacher_chkps) - 1:
            current_index = min(
                GS(s, teacher_chkps, current_index, val_dl, threshold, divergence_metric)
                for s in students
            )
            teacher.load_state_dict(torch.load(teacher_chkps[current_index], map_location=device))
            teacher.eval()
        print(f'Teacher checkpoint index: {current_index}')
        phase = min(n_phase - 1, current_index * n_phase // len(teacher_chkps))
        epoch_losses = []
        for i, (s, m) in enumerate(zip(model_config.keys(), students)):
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
            epoch_losses.append(loss_s)

        ga = EnsembleWeightGA(n_models=len(students))
        ens_weight = ga.evolve(cache.probs.numpy(), labels_all.numpy(), generations=50, patience=8)
        print(f'Epoch {epoch:03d} | loss: {np.mean(epoch_losses):.4f} | ens_weights: {np.round(ens_weight, 3)}')

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

    for batch_x, batch_y, idx in train_dl:
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


'''
TEST BLOCK
'''
@torch.no_grad()
def evaluate(
        model: nn.Module,
        valid_dl: DataLoader,
        loss_fn: nn.Module
) -> dict:
    model.eval()
    predictions, trues, losses = [], [], []
    for item in tqdm(valid_dl, desc='Evaluating', leave=False):
        image = item['image'].to(device)
        label = item['label'].to(device)
        output = model(image)
        losses.append(loss_fn(output, label).item())
        predictions.extend(output.argmax(dim=-1).cpu().tolist())
        trues.extend(label.cpu().tolist())

    return {
        'loss': float(np.mean(losses)),
        'accuracy': accuracy_score(trues, predictions),
        'f1-macro': f1_score(trues, predictions, average='macro'),
    }

def fit_eval(
        model: nn.Module,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        loss_fn: nn.Module,
        epochs: int=10,
        optim: torch.optim.Optimizer=None,
) -> list[dict]:

    history = []
    best_f1 = 0.0

    print('--------START TRAINING--------')
    for i in range(epochs):
        model.train()
        losses = []
        with tqdm(train_dl, desc=f'Epoch {i+1} - Training') as pb:
            for item in pb:
                image = item['image'].to(device)
                label = item['label'].to(device)

                # Forward
                output = model(image)
                loss = loss_fn(output, label)
                losses.append(loss.item())

                # Backward
                optim.zero_grad()
                loss.backward()
                optim.step()

                pb.set_postfix({
                    'loss': np.array(losses).mean()
                })

        scores = evaluate(model, valid_dl, loss_fn)
        scores['train_loss'] = float(np.mean(losses))
        history.append(scores)

        print(
            f'Epoch {i+1}/{epochs} | '
            f'train_loss={scores['train_loss']:.4f} | '
            f'val_loss={scores['loss']:.4f} | '
            f'acc={scores['accuracy']:.4f} | '
            f'f1-macro={scores['f1-macro']:.4f}'
        )

        if scores['f1-macro'] > best_f1:
            best_f1 = scores['f1-macro']
            torch.save(model.state_dict(), './checkpoints/students/student_best.pt')

    return history

def test():
    train_ds = MnistDataset(
        image_path="./data/MNIST/train-images.idx3-ubyte",
        label_path="./data/MNIST/train-labels.idx1-ubyte"
    )

    valid_ds = MnistDataset(
        image_path="./data/MNIST/t10k-images.idx3-ubyte",
        label_path="./data/MNIST/t10k-labels.idx1-ubyte"
    )

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_dl = DataLoader(
        dataset=valid_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    h, w = train_ds[0]['image'].shape[:2]
    c = train_ds[0]['image'].shape[-1] if len(train_ds[0]['image'].shape) == 3 else 1
    lr = 1e-3

    model = Student(
        in_channels=c,
        out_channels=64,
        num_classes=10
    ).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    scores = fit_eval(
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        loss_fn=loss_fn,
        epochs=5,
        optim=optimizer
    )
    print(scores[-1])
'''
END TEST BLOCK
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial on Distillation Frameworks')
    parser.add_argument('--config',
                        dest='config',
                        type=str,
                        help='configuration file',
                        required=True)
    parser.add_argument('--teacher_config',
                        dest='teacher_config',
                        type=str,
                        help='teacher (AASIST) configuration file',
                        default='./config/AASIST.conf')
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        type=str,
        help='output directory for results',
        default='./results'
    )
    parser.add_argument('--seed',
                        type=int,
                        default=21,
                        help='random seed (default: 21)')
    parser.add_argument(
        '--eval',
        action='store_true',
        help='when this flag is given, evaluates given model and exit'
    )
    parser.add_argument('--comment',
                        type=str,
                        default=None,
                        help='comment to describe the saved model')
    parser.add_argument('--eval_model_weights',
                        type=str,
                        default=None,
                        help='directory to the model weight file (can be also given in the config file)')
    main(parser.parse_args())