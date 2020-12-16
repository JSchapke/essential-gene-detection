import os
import sys
sys.path.append('.')
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from utils.utils import set_seed
import tools


class Loss():
    def __init__(self, y):
        self.y = y
        self.pos_mask = y == 1
        self.neg_mask = y == 0

    def __call__(self, out):
        pos_mask = self.pos_mask
        neg_mask = self.neg_mask
        loss_p = F.binary_cross_entropy_with_logits(
            out[pos_mask].squeeze(), self.y[self.pos_mask].cuda())
        loss_n = F.binary_cross_entropy_with_logits(
            out[neg_mask].squeeze(), self.y[neg_mask].cuda())
        loss = loss_p + loss_n
        return loss


def acc(t1, t2):
    return np.sum(t1*1 == t2*1) / len(t1)


def mlp_fit_predict(train_x, train_y, test_x, val=None, return_val_probs=False):
    epochs = 1000

    in_feats = train_x.shape[1]
    model = nn.Sequential(
        nn.Linear(in_feats, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters())

    lossf = Loss(train_y)

    if val is not None:
        val_x, val_y = val
        lossf_val = Loss(val_y)

    model.train()
    model.cuda()

    patience, cur_es = 3, 0
    val_loss_old = np.Inf

    for i in range(epochs):
        out = model(train_x)
        loss = lossf(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % 10) == 0:
            if val is not None:
                model.eval()
                with torch.no_grad():
                    loss_val = lossf_val(model(val_x))
                print(f'{i}. Train loss:', loss.detach().cpu().numpy(), ' |  Val Loss:', loss_val.detach().cpu().numpy())
                model.train()

                if val_loss_old < loss_val:
                    cur_es += 1
                else:
                    cur_es = 0
                val_loss_old = loss_val

                if cur_es == patience:
                    break

    model.eval()
    with torch.no_grad():
        out = model(test_x).cpu()
    probs = torch.sigmoid(out).numpy()

    if return_val_probs:
        with torch.no_grad():
            out = model(val_x).cpu()
        val_probs = torch.sigmoid(out).numpy()

        return probs, val_probs

    return probs


def main(args):
    roc_aucs = []
    for i in range(args.n_runs):
        seed = i
        set_seed(seed)

        _, X, (train_idx, train_y), (val_idx, val_y), (test_idx,
                                                       test_y), names = tools.get_data(args.__dict__, seed=seed)

        if X is None or not X.shape[1]:
            raise ValueError('No features')

        train_x = X[train_idx].cuda()
        val_x = X[val_idx].cuda()
        test_x = X[test_idx].cuda()
        print('train_x', train_x.mean())
        print('test_x', test_x.mean())

        probs = mlp_fit_predict(train_x, train_y, test_x, val=(val_x, val_y))
        roc_auc = roc_auc_score(test_y, probs)
        roc_aucs.append(roc_auc)

        p = np.concatenate(
            [names[test_idx].reshape(-1, 1), probs.reshape(-1, 1)], axis=1)
        save_preds(p, args, seed)

    print('Auc(all):', roc_aucs)
    print('Auc:', np.mean(roc_aucs))

    return np.mean(roc_aucs), np.std(roc_aucs)


def save_preds(preds, args, seed):
    name = get_name(args) + f'_{args.organism}_{args.ppi}_s{seed}.csv'
    name = name.lower()
    path = os.path.join('outputs/preds/', name)
    df = pd.DataFrame(preds, columns=['Gene', 'Pred'])
    df.to_csv(path)
    print('Saved the predictions to:', path)


def get_name(args):
    if args.name:
        return args.name

    name = 'MLP'
    if args.no_ppi:
        name += '_NO-PPI'
    if args.expression:
        name += '_EXP'
    if args.sublocs:
        name += '_SUB'
    if args.orthologs:
        name += '_ORT'

    return name


if __name__ == '__main__':
    args = tools.get_args()

    mean, std = main(args)

    name = get_name(args)

    df_path = 'outputs/results/results.csv'
    df = pd.read_csv(df_path)

    df.loc[len(df)] = [name, args.organism, args.ppi, args.expression,
                       args.orthologs, args.sublocs, args.n_runs, mean, std]
    df.to_csv(df_path, index=False)
    print(df.head())
