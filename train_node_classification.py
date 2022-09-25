from __future__ import division
import argparse
import time
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('.')
from utility.cora_dataset import get_planetoid_dataset
from utility import utils, psgd
import itertools
from models.node_cls_models import NodeClsESGNN, GCN


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--model', type=str, default='esn', choices=['esn', 'gcn'])
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--split', type=str, default='complete', choices=['public', 'full', 'complete'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.005)   # 0.005
parser.add_argument('--early_stopping', type=int, default=0)
parser.add_argument('--hidden', type=int, default=1000)
parser.add_argument('--dropout', type=float, default=0.2)   # 0.2
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--logger', type=str, default=None)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--preconditioner', type=str, default=None)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--eps', type=float, default=0.001)
parser.add_argument('--update_freq', type=int, default=50)
parser.add_argument('--gamma', type=float, default=None)
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--hyperparam', type=str, default=None, choices=['eps', 'update_freq', 'gamma', None])
parser.add_argument('--n_iter', type=int, default=2)
parser.add_argument('--sparsity', type=float, default=0.4, help='weight sparsity') #0.4
parser.add_argument('--weight_dist', type=str, default='gaussian', choices=['uniform', 'gaussian'])
parser.add_argument('--hid_init_dist', type=str, default='gaussian', choices=['uniform', 'gaussian'])
parser.add_argument('--hid_init_scale', type=int, default=1.)

parser.add_argument('--posreg', type=str, default=False, help='postive weight regularization')
parser.add_argument('--save_model', type=str, default=False, help='save models')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path_runs = "runs"
save_dir = 'checkpoints'
for dir in [path_runs, save_dir]:
    if not os.path.exists(dir):
        os.mkdir(dir)


def run(
        model,
        str_optimizer,
        str_preconditioner,
        epochs,
        lr,
        weight_decay,
        early_stopping,
        logger,
        momentum,
        eps,
        update_freq,
        gamma,
        alpha,
        hyperparam,
        noise,
        save_model=False,
):

    if logger is not None:
        if hyperparam:
            logger += f"-{hyperparam}{eval(hyperparam)}"
        path_logger = os.path.join(path_runs, logger)

        utils.empty_dir(path_logger)
        logger = SummaryWriter(log_dir=os.path.join(
            path_runs, logger)) if logger is not None else None

    train_losses, val_losses, accs, durations = [], [], [], []
    model.to(device)

    if str_preconditioner == 'KFAC':

        preconditioner = psgd.KFAC(
            model,
            eps,
            sua=False,
            pi=False,
            update_freq=update_freq,
            alpha=alpha if alpha is not None else 1.,
            constraint_norm=False
        )
    else:
        preconditioner = None

    if str_optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif str_optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    best_val_loss = float('inf')
    best_te_acc = 0

    train_loss = 0
    val_loss_history = []

    for epoch in range(1, epochs + 1):
        lam = (float(epoch)/float(epochs)
                )**gamma if gamma is not None else 0.
        # train
        out = train(model, optimizer, data, preconditioner, lam)
        eval_info = evaluate(model, data)

        eval_info['epoch'] = int(epoch)
        eval_info['time'] = time.perf_counter() - t_start
        eval_info['eps'] = eps
        eval_info['update-freq'] = update_freq

        if gamma is not None:
            eval_info['gamma'] = gamma

        if alpha is not None:
            eval_info['alpha'] = alpha

        if logger is not None:
            for k, v in eval_info.items():
                logger.add_scalar(k, v, global_step=epoch)
            for name, w in model.named_parameters():
                logger.add_histogram(name, w)

        if eval_info['val loss'] < best_val_loss:
            train_loss = eval_info['train loss']
            best_val_loss = eval_info['val loss']
            best_te_acc = eval_info['test acc']

        val_loss_history.append(eval_info['val loss'])
        if early_stopping > 0 and epoch > epochs // 2:
            tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
            if eval_info['val loss'] > tmp.mean().item():
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()

    val_losses.append(best_val_loss)
    train_losses.append(train_loss)
    accs.append(best_te_acc)
    durations.append(t_end - t_start)

    if logger is not None:
        logger.close()
    tr_loss, loss, acc, duration = tensor(train_losses), tensor(val_losses), tensor(accs), tensor(durations)
    print(f'Tr Loss: {tr_loss.mean().item():.4f}, Val Loss: {loss.mean().item():.4f}, '
          f'Test Accuracy: {100*acc.mean().item():.2f}%, Duration: {duration.mean().item():.3f}s.'
                 )

    model_filename = utils.filename_with_time(f'cora_acc{acc.mean().item() * 10000:.0f}_noise_{noise}.pt', save_dir)
    if save_model:
        torch.save({'model': model,
                    'acc': acc},
                    model_filename)

    return loss, acc, duration


def train(model, optimizer, data, preconditioner=None, lam=0.):
    model.train()
    optimizer.zero_grad()
    out, node_embeddings = model(data)

    label = out.max(1)[1]
    label[data.train_mask] = data.y[data.train_mask]
    label.requires_grad = False

    loss = F.cross_entropy(out[data.train_mask], label[data.train_mask])
    loss.backward(retain_graph=True)

    optimizer.step()

    return out


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits, _ = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{} loss'.format(key)] = loss
        outs['{} acc'.format(key)] = acc
        outs['{} prediction'.format(key)] = pred

    return outs

'''Load dataset'''
dataset = get_planetoid_dataset(name=args.dataset, normalize_features=args.normalize_features, split=args.split)
data = dataset[0].to(device, 'x', 'y')

if __name__ == '__main__':

    # Build model and train
    test_accs = []
    if args.model == 'gcn':
        model = GCN(dataset, args.hidden, args.dropout)
    elif args.model == 'esn':
        model = NodeClsESGNN(dataset, args.hidden, args.n_iter, args.sparsity, weight_dist=args.weight_dist,
                                    hid_init_dist='zero', hid_init_scale=1, noise=args.noise)

    model.to(device)

    kwargs = {
        'model': model,
        'str_optimizer': args.optimizer,
        'str_preconditioner': args.preconditioner,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'early_stopping': args.early_stopping,
        'logger': args.logger,
        'momentum': args.momentum,
        'eps': args.eps,
        'update_freq': args.update_freq,
        'gamma': args.gamma,
        'alpha': args.alpha,
        'hyperparam': args.hyperparam,
        'noise': args.noise,
        'save_model': args.save_model
    }

    if args.hyperparam == 'eps':
        for param in np.logspace(-3, 0, 10, endpoint=True):
            print(f"{args.hyperparam}: {param}")
            kwargs[args.hyperparam] = param
            _, _, _ = run(**kwargs)
    elif args.hyperparam == 'update_freq':
        for param in [4, 8, 16, 32, 64, 128]:
            print(f"{args.hyperparam}: {param}")
            kwargs[args.hyperparam] = param
            _, _, _ = run(**kwargs)
    elif args.hyperparam == 'gamma':
        for param in np.linspace(1., 10., 10, endpoint=True):
            print(f"{args.hyperparam}: {param}")
            kwargs[args.hyperparam] = param
            _, _, _ = run(**kwargs)
    else:
        _, test_acc, _ = run(**kwargs)
