'''
Methods based on formulas
'''
import os
import argparse
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--organism', default='yeast')
parser.add_argument('--ppi', default='biogrid')
parser.add_argument('--string_thr', default=500, type=int,
                    help='Connection threshold for STRING PPI database')
parser.add_argument('--n_runs', default=1, type=int)
args = parser.parse_args()

# Neighborhood based methods


def DC(G, test_X):
    '''
    Degree Centrality
    '''
    preds = []
    n_nodes = len(G.nodes())
    for x in test_X:
        if x not in G:
            preds.append(0)
            continue
        nghbrs = len(list(G.neighbors(x)))
        score = nghbrs / (n_nodes - 1)
        preds.append(score)
    preds = np.stack([test_X, preds], axis=1)
    return preds


def LAC(G, test_X):
    '''
    Local Average Connectivity
    '''
    preds = []
    for node in test_X:
        if node not in G:
            preds.append(0)
            continue
        neigs = list(G.neighbors(node))
        subG = G.subgraph(neigs)
        score = 0

        for neig in neigs:
            neig_deg = subG.degree(neig)
            score += neig_deg
        score /= len(neigs) if len(neigs) else 1

        preds.append(score)
    preds = np.stack([test_X, preds], axis=1)
    return preds


def NC(G, test_X):
    '''
    Edge clustering coefficient centrality
    '''
    preds = []
    for node in test_X:
        if node not in G:
            preds.append(0)
            continue
        node_degree = G.degree(node)
        score = 0
        neigs = np.array(list(G.neighbors(node)))
        for neig in neigs:
            neig_degree = G.degree(neig)
            neig_neigs = np.array(list(G.neighbors(neig)))
            intersect = np.intersect1d(neigs, neig_neigs)

            z = len(intersect)
            min_degree = min(node_degree-1, neig_degree-1)
            if min_degree != 0:
                score += z / min_degree
        preds.append(score)
    preds = np.stack([test_X, preds], axis=1)
    return preds


def align(test, probs):
    preds, y = [], []
    for prob in probs:
        mask = prob[0] == test[:, 0]
        if np.any(mask):
            preds.append(prob[1])
            y.append(test[mask][0, 1])
    preds = np.array(preds, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return y, preds


def plot_results(root, test):
    for f in os.listdir(root):
        if not f.endswith('.npy'):
            continue
        path = os.path.join(root, f)
        probs = np.load(path, allow_pickle=True)
        y, preds = align(test, probs)
        fpr, tpr, _ = roc_curve(y, preds)
        roc_auc = roc_auc_score(y, preds)

        lw = 2
        plt.plot(fpr, tpr, lw=lw, label=f'{f[:-4]} (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Results on Essential Genes dataset')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(root, 'results.png'))


def test_auc(test, preds):
    y, preds = align(test, preds)
    fpr, tpr, _ = roc_curve(y, preds)
    roc_auc = roc_auc_score(y, preds)
    return roc_auc

def save_preds(preds, name, args):
    name = name + f'_{args.organism}_{args.ppi}.csv'
    name = name.lower()
    path = os.path.join('preds', name)
    df = pd.DataFrame(preds, columns=['Gene', 'Pred'])
    df.to_csv(path)
    print('Saved the predictions to:', path)


def run(seed=0, save=True, update=True):
    # Getting the data ----------------------------------
    (edges, _), _, _, test, genes = utils.data(
        organism=args.organism, ppi=args.ppi, seed=seed)

    G = nx.Graph()
    G.add_edges_from(edges)

    test_X = test[:, 0]

    DC_results = DC(G, test_X)
    DC_auc = test_auc(test, DC_results)
    save_preds(DC_results, 'DC', args)

    LAC_results = LAC(G, test_X)
    LAC_auc = test_auc(test, LAC_results)
    save_preds(LAC_results, 'LAC', args)

    NC_results = NC(G, test_X)
    NC_auc = test_auc(test, NC_results)
    save_preds(NC_results, 'NC', args)

    return DC_auc, LAC_auc, NC_auc


def write_results(path, DC_aucs, LAC_aucs, NC_aucs):
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=['Model Type', 'Organism', 'PPI', 'Expression',
                                   'Orthologs', 'Sublocalization', 'N Runs', 'Mean', 'Std Dev'])

    df.loc[len(df)] = ['DC', args.organism, args.ppi, '', '', '',
                       args.n_runs, np.mean(DC_aucs), np.std(DC_aucs)]
    df.loc[len(df)] = ['LAC', args.organism, args.ppi, '', '', '',
                       args.n_runs, np.mean(LAC_aucs), np.std(LAC_aucs)]
    df.loc[len(df)] = ['NC', args.organism, args.ppi, '', '', '',
                       args.n_runs, np.mean(NC_aucs), np.std(NC_aucs)]
    df.to_csv(path, index=False)


if __name__ == '__main__':
    DC_aucs = []
    LAC_aucs = []
    NC_aucs = []

    for i in range(args.n_runs):
        dc, lac, nc = run(seed=i, save=False)
        DC_aucs.append(dc)
        LAC_aucs.append(lac)
        NC_aucs.append(nc)

    path1 = 'results/results.csv'
    write_results(path1, DC_aucs, LAC_aucs, NC_aucs)
