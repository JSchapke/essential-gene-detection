import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import sklearn
import sklearn.svm
from utils.utils import *

import tools


def acc(t1, t2):
    return np.sum(1.0*(t1 == t2)) / len(t1)


def main(args):
    roc_aucs = []
    for i in range(args.n_runs):
        seed = i
        set_seed(seed)

        (_, _), X, (train_idx, train_y), (val_idx, val_y), (test_idx,
                                                            test_y), names = tools.get_data(args.__dict__, seed=seed)
        if X is None or not X.shape[1]:
            raise ValueError('No features')

        clf = sklearn.svm.SVC(class_weight='balanced', random_state=seed, probability=True)
        clf.fit(X[train_idx], train_y)
        probs = clf.predict_proba(X[test_idx])[:, 1]
        roc_auc = roc_auc_score(test_y, probs)
        roc_aucs.append(roc_auc)

        p = np.stack([names[test_idx], probs], axis=1)
        save_preds(p, args, seed)

    print('Auc(all):', roc_aucs)
    print('Auc:', np.mean(roc_aucs))

    return np.mean(roc_aucs), np.std(roc_aucs)


def get_name(args):
    if args.name:
        return args.name

    name = 'SVM'
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
    name = get_name(args) + f'_{args.organism}_{args.ppi}_s{seed}.csv'
    name = name.lower()
    path = os.path.join('preds', name)
    df = pd.DataFrame(preds, columns=['Gene', 'Pred'])
    df.to_csv(path)
    print('Saved the predictions to:', path)


def write(path):
    df = pd.read_csv(path)
    df.loc[len(df)] = [name, args.organism, args.ppi, args.expression,
                       args.orthologs, args.sublocs, args.n_runs, mean, std]
    df.to_csv(path, index=False)
    print(df.head())


if __name__ == '__main__':
    args = tools.get_args()

    mean, std = main(args)

    name = get_name(args)

    path1 = 'results/results.csv'
    path2 = f'results/{args.organism}.csv'
    write(path1)
    write(path2)
