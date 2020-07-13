import os
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from tqdm import tqdm
import optuna
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim

from models.gat.gat_pytorch import GAT
from models.gat import params as gat_params
from utils.utils import *
from runners import tools

GAT_P1 = gat_params.gat_0
GAT_P2 = gat_params.gat_fly
GAT_P3 = gat_params.gat_yeast
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def optimize(X, A, train_y, train_idx, val_y, val_idx):
    def objective(trial):
        n_layers = trial.uniform('n_layers', 1, 4)
        h_feats = [trial.uniform(f'h_feat_{i}', 4, 32)
                   for i in range(n_layers)]
        heads = [trial.uniform(f'head_{i}', 1, 12) for i in range(n_layers)]

        params = {
            'lr': trial.loguniform('lr', 5e-3, 1e-5),
            'weight_decay': trial.loguniform('weight_decay', 5e-4, 1e-6),
            'h_feats': h_feats,
            'heads': heads,
            'dropout': trial.uniform('dropout', 0.02, 0.5),
            'negative_slope': 0.2}

        score, _ = train(X, A,
                         train_y, train_idx,
                         val_y, val_idx,
                         params=params,
                         return_score=True)

    study = optuna.create_study(objective, trials=200)
    study.optimize()
    best_params = study.best_params
    val_auc = study.w


def train(params, X, A,
          edge_weights,
          train_y, train_idx,
          val_y, val_idx,
          return_score=False,
          save_best_only=True,
          savepath='',
          ):

    epochs = 1000

    model = GAT(in_feats=X.shape[1], **params)
    model.to(DEVICE)
    X = X.to(DEVICE)
    A = A.to(DEVICE)
    train_y = train_y.to(DEVICE)
    val_y = val_y.to(DEVICE)
    if edge_weights is not None:
        edge_weights = edge_weights.to(DEVICE)

    wa = tools.WeightAveraging(model, epochs-500, 100)
    optimizer = optim.Adam(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fnc = tools.Loss(train_y, train_idx)
    val_loss_fnc = tools.Loss(val_y, val_idx)

    iterable = tqdm(range(epochs))
    for i in iterable:
        model.train()
        logits = model(X, A, edge_weights=edge_weights)

        optimizer.zero_grad()
        loss = loss_fnc(logits)
        loss.backward()
        optimizer.step()
        wa.step()

        logits = logits.detach()
        val_loss = val_loss_fnc(logits)
        train_auc = evalAUC(None, 0, 0, train_y, 0, logits[train_idx])
        val_auc = evalAUC(None, 0, 0, val_y, 0, logits[val_idx])

        tqdm.set_description(iterable, desc='Loss: %.4f ; Val Loss %.4f ; Train AUC %.4f. Validation AUC: %.4f' % (
            loss, val_loss, train_auc, val_auc))

    wa.set_weights()
    score = evalAUC(model, X, A, val_y, val_idx)
    print(f'Last validation AUC: {val_auc}')
    print(f'WA validation AUC: {score}')

    if savepath:
        save = {
            'auc': score,
            'model_params': params,
            'model_state_dict': model.state_dict()
        }
        torch.save(save, savepath)

    if return_score:
        return score, model
    return model


def test(model, X, A, test_ds=None):
    model.to(DEVICE).eval()
    X = X.to(DEVICE)
    A = A.to(DEVICE)

    with torch.no_grad():
        logits = model(X, A)
    probs = torch.sigmoid(logits)
    probs = probs.cpu().numpy()

    if test_ds is not None:
        test_idx, test_y = test_ds
        test_y = test_y.cpu().numpy()
        auc = metrics.roc_auc_score(test_y, probs[test_idx])
        return probs, auc
    return probs


def get_params(org):
    if org == 'melanogaster':
        return GAT_P2
    elif org == 'yeast':
        return GAT_P3
    else:
        return GAT_P1


def get_name(args):
    if args.name:
        return args.name
    name = 'GAT'
    if args.no_ppi:
        name += '_NO-PPI'
    if args.expression:
        name += '_EXP'
    if args.sublocs:
        name += '_SUB'
    if args.orthologs:
        name += '_ORT'
    return name


def save_preds(preds, name, args, seed):
    name = name.lower() + f'_{args.organism}_{args.ppi}_s{seed}.csv'
    path = os.path.join('preds', name)

    df = pd.DataFrame(preds, columns=['Gene', 'Pred'])
    df.to_csv(path)
    print('Saved the predictions to:', path)


def main(args, name='', seed=0, save=True):
    set_seed(seed)

    snapshot_name = f'{args.organism}_{args.ppi}'
    for p in ['expression', 'orthologs', 'sublocs']:
        if args.__dict__[p]:
            snapshot_name += f'_{p}'

    weightsdir = './models/gat/weights'
    outdir = './results/{args.organism}/gat'
    savepath = os.path.join(weightsdir, snapshot_name)

    # Getting the data ----------------------------------
    (edge_index, edge_weights), X, (train_idx, train_y), \
        (val_idx, val_y), (test_idx, test_y), genes = tools.get_data(
            args.__dict__, seed=seed, weights=False)
    print('Fetched data')
    # ---------------------------------------------------

    # Train the model -----------------------------------
    if args.train:
        print('\nTraining the model')
        gat_params = get_params(args.organism)
        model = train(gat_params, X, edge_index, edge_weights,
                      train_y, train_idx, val_y, val_idx, savepath=savepath)
    # ---------------------------------------------------

    # Load trained model --------------------------------
    print(f'\nLoading the model from: {savepath}')
    snapshot = torch.load(savepath)
    model = GAT(in_feats=X.shape[1], **snapshot['model_params'])
    model.load_state_dict(snapshot['model_state_dict'])
    print('Model loaded. Val AUC: {}'.format(snapshot['auc']))
    # ---------------------------------------------------

    # Test the model ------------------------------------
    preds, auc = test(model, X, edge_index, (test_idx, test_y))
    preds = np.concatenate(
        [genes[test_idx].reshape((-1, 1)), preds[test_idx]], axis=1)
    print('Test AUC:', auc)
    # ---------------------------------------------------

    return preds, auc


if __name__ == '__main__':
    args = tools.get_args()

    name = get_name(args)

    if args.n_runs:
        args.train = True
        args.test = True

        scores = []
        for i in range(args.n_runs):
            preds, auc = main(args, name=name, seed=i)
            scores.append(auc)

            save_preds(preds, name, args, seed=i)

        mean = np.mean(scores)
        std = np.std(scores)

        df_path = 'results/results.csv'
        try:
            df = pd.read_csv(df_path)
        except:
            df = pd.DataFrame([], columns=['Model Type', 'Organism', 'PPI', 'Expression',
                                           'Orthologs', 'Sublocalization', 'N Runs', 'Mean', 'Std Dev'])
        df.loc[len(df)] = [name, args.organism, args.ppi, args.expression,
                           args.orthologs, args.sublocs, args.n_runs, mean, std]
        df.to_csv(df_path, index=False)

        df_path = f'results/{args.organism}.csv'
        try:
            df = pd.read_csv(df_path)
        except:
            df = pd.DataFrame([], columns=[
                              'name', 'ppi', 'expression', 'orthologs', 'sublocs', 'n_runs', 'mean', 'std'])
        df.loc[len(df)] = [name, args.ppi, args.expression,
                           args.orthologs, args.sublocs, args.n_runs, mean, std]
        df.to_csv(df_path, index=False)

        print('Final Result:', mean)
        print(df.tail())

    else:
        main(args, name=name, seed=0)
