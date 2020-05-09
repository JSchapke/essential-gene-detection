import sys; sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import sklearn
import sklearn.ensemble
from utils import *

import tools

def acc(t1, t2):
    return np.sum(1.0*(t1==t2)) / len(t1)

def main(args):
    scores, roc_aucs = [], []
    for i in range(5):
        seed = i
        set_seed(seed)
        
        (_, _), X, (train_idx, train_y), (val_idx, val_y), (test_idx, test_y), names = tools.get_data(args.__dict__, seed=seed)
        if X is None or not X.shape[1]:
            raise ValueError('No features')

        clf = sklearn.ensemble.RandomForestClassifier(class_weight='balanced', random_state=seed, n_estimators=500)
        clf.fit(X[train_idx], train_y)
        probs = clf.predict(X[test_idx])

        roc_auc = roc_auc_score(test_y, probs)
        roc_aucs.append(roc_auc)

        preds = (probs > 0.5) * 1
        score = acc(preds, test_y)
        print('Score:', score)
        scores.append(score)

    print('Acc(all):', scores)
    print('Auc(all):', roc_aucs)
    print('Accuracy:', np.mean(scores))
    print('Auc:', np.mean(roc_aucs))

    return np.mean(roc_aucs), np.std(roc_aucs)

def get_name(args):
    if args.name:
        return args.name

    name = 'RandomForest'
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


    df_path = 'results/results.csv'
    df = pd.read_csv(df_path)

    df.loc[len(df)] = [name, args.organism, args.ppi, args.expression, args.orthologs, args.sublocs, args.n_runs, mean, std]
    df.to_csv(df_path, index=False)
    print(df.head())
