import os
import argparse
import sys; sys.path.append('.')
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from utils import *
import random

parser = argparse.ArgumentParser()
parser.add_argument('--organism', default='yeast')
parser.add_argument('--ppi', default='biogrid')
parser.add_argument('--all_gats', action='store_true')
parser.add_argument('--gat_best', action='store_true')
parser.add_argument('--gat_ppi', action='store_true')
parser.add_argument('--methods', action='store_true')
args = parser.parse_args()

def plot_results(paths, test):
    for path in paths:
        if not path.endswith('.npy'):
            continue

        probs = np.load(path, allow_pickle=True)
        f = os.path.basename(path)
        print(f[:-4], probs.shape, test.shape)

        preds, y = [], []
        for prob in probs:
            mask = prob[0] == test[:, 0]
            if np.any(mask):
                preds.append(prob[1])
                y.append(test[mask][0, 1])
        preds = np.array(preds, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        fpr, tpr, _ = roc_curve(y, preds) 
        roc_auc = roc_auc_score(y, preds)

        lw = 2
        plt.plot(fpr, tpr, lw=lw, label=f'{f[:-4]} (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Performance on {args.organism} with the {args.ppi} network.')
    plt.legend(loc="lower right")


if __name__ == '__main__':

    ROOT = '/home/schapke/projects/research/2020_FEB_JUN/src/results/'
    path = os.path.join(ROOT, args.organism, args.ppi)
    paths = []

    if args.all_gats:
        print('--plotting GATs')
        p = os.path.join(path, 'gat')
        files = os.listdir(p)
        paths += [os.path.join(p, f) for f in files]
    else:
        if args.gat_best:
            print('--plotting GAT*')
            paths.append(os.path.join(path, 'gat/GAT*.npy'))

        if args.gat_ppi:
            print('--plotting GAT')
            paths.append(os.path.join(path, f'gat/GAT.npy'))

    if args.methods:
        print('--plotting methods')
        p = os.path.join(path, 'methods')
        files = os.listdir(p)
        paths += [os.path.join(p, f) for f in files]


    if not len(paths):
        raise Exception('No arguments especified')

    if args.organism == 'yeast':
        G, X, train, test = yeast_data(ppi=args.ppi)
    elif args.organism == 'coli':
        G, X, train, test = coli_data(ppi=args.ppi)
    elif args.organism == 'human':
        G, X, train, test = human_data(ppi=args.ppi)


    plot_results(paths, test)

    plot_name = f'{args.organism}_{args.ppi}'
    if args.methods:
        plot_name += '_methods'
    if args.all_gats:
        plot_name += '_gats'
    if args.gat_ppi or args.gat_best:
        plot_name += '_gat'
    plot_name += '.png'
    plt.savefig(os.path.join('plots', plot_name))

