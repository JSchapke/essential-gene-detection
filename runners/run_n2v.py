import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from torch_geometric.nn.models import Node2Vec
from sklearn.metrics import roc_auc_score
import os
import sys


PARAMS = {
    'embedding_dim': 128,
    'walk_length': 64,
    'context_size': 64,
    'walks_per_node': 64,
    'num_negative_samples': 1,
}
LR = 1e-2
WEIGHT_DECAY = 5e-4
EPOCHS = 50
DEV = torch.device('cuda')


def train_epoch(n2v, n2v_loader, n2v_optimizer, X, train_y, train_mask, val_y, val_mask, test_mask):
    print('Two-Step model train epoch')

    X = X.to(DEV)
    train_y = train_y.to(DEV)
    val_y = val_y.to(DEV)
    Z = None

    n2v.train()
    for i in range(EPOCHS):
        n2v_train_loss = 0

        for (pos_rw, neg_rw) in n2v_loader:
            n2v_optimizer.zero_grad()
            loss = n2v.loss(pos_rw.to(DEV), neg_rw.to(DEV))
            loss.backward()
            n2v_optimizer.step()
            n2v_train_loss += loss.data.item()
        print(f'Epoch {i}. N2V Train_Loss:', n2v_train_loss)
    print('')
    n2v.eval()
    Z = n2v().detach()

    if X is None:
        train_x = Z[train_mask]
        val_x = Z[val_mask]
        test_x = Z[test_mask]
    elif Z is not None:
        train_x = torch.cat([Z[train_mask], X[train_mask]], dim=1)
        val_x = torch.cat([Z[val_mask], X[val_mask]], dim=1)
        test_x = torch.cat([Z[test_mask], X[test_mask]], dim=1)
    else:
        train_x = X[train_mask]
        val_x = X[val_mask]
        test_x = X[test_mask]
    print('train_X.shape', train_x.shape)

    probs, val_probs = mlp_fit_predict(
        train_x, train_y, test_x, val=(val_x, val_y), return_val_probs=True)
    val_roc_auc = roc_auc_score(val_y.cpu().numpy(), val_probs)

    print('Validation ROC_AUC:', val_roc_auc)
    return probs, val_roc_auc


def fit_predict(edge_index, X, train_y, train_mask, val_y, val_mask, test_mask):
    n2v = Node2Vec(edge_index, **PARAMS).to(DEV)
    n2v_loader = n2v.loader(batch_size=128, shuffle=True, num_workers=0)
    n2v_optimizer = optim.Adam(n2v.parameters(), lr=LR)

    probs, val_roc_auc = train_epoch(
        n2v, n2v_loader, n2v_optimizer, X, train_y, train_mask, val_y, val_mask, test_mask)

    return probs


def main(args):
    roc_aucs = []
    for i in range(max(args.n_runs, 1)):
        seed = i
        set_seed(seed)

        (edge_index, _), X, (train_idx, train_y), \
            (val_idx, val_y), (test_idx, test_y), names = tools.get_data(
                args.__dict__, seed=seed)

        if X is None or not X.shape[1]:
            raise ValueError('No features')

        probs = fit_predict(edge_index, X, train_y,
                            train_idx, val_y, val_idx, test_idx)
        auc = roc_auc_score(test_y.cpu().numpy(), probs)
        roc_aucs.append(auc)
        print('Final AUC:', auc)

        test_genes = names[test_idx].reshape(-1, 1)
        test_y = test_y.reshape(-1, 1)
        preds = np.concatenate([probs, test_genes, test_y], axis=1)
        save_preds(preds, args, seed)

    print('Auc(all):', roc_aucs)
    print('Auc:', np.mean(roc_aucs))
    return np.mean(roc_aucs), np.std(roc_aucs)


def get_name(args):
    if args.name:
        return args.name

    name = 'N2V'
    if args.no_ppi:
        name += '_NO-PPI'
    if args.expression:
        name += '_EXP'
    if args.sublocs:
        name += '_SUB'
    if args.orthologs:
        name += '_ORT'

    return name


def save_preds(preds, args, seed):
    os.makedirs("./preds", exist_ok=True)
    name = get_name(args) + f'_{args.organism}_{args.ppi}_s{seed}.csv'
    name = name.lower()
    path = os.path.join('preds', name)
    df = pd.DataFrame(preds, columns=['Pred', 'Gene', 'Label'])
    df.to_csv(path)
    print('Saved the predictions to:', path)


if __name__ == '__main__':
    sys.path.append('.')
    from runners import tools
    from runners.run_mlp import mlp_fit_predict
    from utils.utils import *

    args = tools.get_args(parse=True)
    print("ARGS:", args)

    mean, std = main(args)

    name = get_name(args)

    df_path = 'results/results.csv'
    if os.path.isfile(df_path):
        df = pd.read_csv(df_path)
    else:
        df = pd.DataFrame([], columns=["name", "organism", "ppi",
                                       "expression", "orthology", "subloc", "n_runs", "mean", "std"])
        os.makedirs('results', exist_ok=True)
    df.loc[len(df)] = [name, args.organism, args.ppi, args.expression,
                       args.orthologs, args.sublocs, args.n_runs, mean, std]
    df.to_csv(df_path, index=False)
    print(df.tail())
