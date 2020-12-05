import os
import random
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy.sparse as sparse
from os.path import join
import pickle
from tqdm import tqdm
DATA_ROOT = '../data'


def evalAUC(model, X, A, y, mask, logits=None):
    assert(model is not None or logits is not None)
    if model is not None:
        model.eval()
        with torch.no_grad():
            logits = model(X, A)
            logits = logits[mask]
    probs = torch.sigmoid(logits)
    probs = probs.cpu().numpy()
    y = y.cpu().numpy()
    auc = metrics.roc_auc_score(y, probs)
    return auc


def evaluate(model, X, A, y, mask):
    model.eval()
    with torch.no_grad():
        logits = model(X, A)
        logits = logits[mask]
        preds = (torch.sigmoid(logits) > 0.5).to(torch.float32)[:, 0]
        correct = torch.sum(preds == y)
        return correct.item() * 1.0 / len(y)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dim_reduction_cor(X, y, k=20):
    cors = np.zeros((X.shape[1])) 

    # calculate the correlation with y for each feature
    for i in range(X.shape[1]):
        cor = np.corrcoef(X[:, i], y)[0, 1]
        if not np.isnan(cor):
            cors[i] = cor

    features = np.zeros_like(cors).astype(bool)
    features[np.argsort(-cors)[:k]] = True

    return features, cors

def filter_labels(labels, genes):
    mask = np.isin(labels[:, 0], genes)
    return labels[mask]

def data(
    organism='yeast',
    ppi='string', 
    expression=False, 
    orthologs=False, 
    sublocalizations=False,
    string_thr=500, 
    seed=0, 
    weights=False):

    assert(organism in ['yeast', 'coli', 'human', 'melanogaster'])
    assert(organism not in ['coli'] or not sublocalizations)
    assert(ppi in ['string', 'dip', 'biogrid'])

    print(f'\nGathering {organism} dataset.')
    print(f'PPI: {ppi}.')

    # Cache ----------
    update = True
    cache = f'.cache/{organism}/'
    cachepath = cache + f'{expression}_{sublocalizations}_{orthologs}_{ppi}.pkl'
    os.makedirs(cache, exist_ok=True)

    if os.path.isfile(cachepath) and not update:
        print('Data was cached')
        with open(cachepath, 'rb') as f:
            edges, edge_weights, X, labels, genes = pickle.load(f)

    else:
        edge_weights = None

        if ppi in ['biogrid', 'dip']:
            ppi_path = os.path.join(DATA_ROOT, f'essential_genes/{organism}/PPI/{ppi.upper()}/{ppi}.csv')
            edges = pd.read_csv(ppi_path)

        elif ppi == 'string':
            ppi_path = os.path.join(DATA_ROOT, f'essential_genes/{organism}/PPI/STRING/string.csv')
            edges = pd.read_csv(ppi_path)

            key = 'combined_score'
            edges = edges[edges.loc[:, key] > string_thr].reset_index()

            edge_weights = edges['combined_score'] / 1000
            edges = edges[['A', 'B']]
            print('Filtered String network with thresh:', string_thr)


        edges = edges.dropna()
        index, edges = edges.index, edges.values
        ppi_genes = np.union1d(edges[:, 0], edges[:, 1])
        if edge_weights is not None: 
            edge_weights = edge_weights.iloc[index.values].values

        path = os.path.join(DATA_ROOT, f'essential_genes/{organism}/EssentialGenes/ogee.csv')
        labels = pd.read_csv(path).values

        # filter labels not in the PPI network
        print('Number of labels before filtering:', len(labels))
        labels = filter_labels(labels, ppi_genes)
        print('Number of labels after filtering:', len(labels))

        genes = np.union1d(labels[:, 0], ppi_genes)
        print('Total number of genes:', len(genes))

        X = np.zeros((len(genes), 0))

        if orthologs:
            path = os.path.join(DATA_ROOT, f'essential_genes/{organism}/Orthologs/orthologs.csv')
            orths = pd.read_csv(path)
            print('Orthologs dataset shape:', orths.shape)

            x = np.zeros((len(genes), orths.shape[1]-1))
            for i, node in enumerate(genes):
                mask = orths['Gene'] == node
                if np.any(mask):
                    x[i] = orths[mask].values.squeeze()[:-1]
            X = np.concatenate([X, np.array(x)], axis=1)

        if expression:
            if organism == 'human':
                path = os.path.join(DATA_ROOT, 'essential_genes/human/Expression/expression_64.csv')
            else:
                path = os.path.join(DATA_ROOT, f'essential_genes/{organism}/Expression/profile.csv')
            expression = pd.read_csv(path).values
            print('Gene expression dataset shape:', expression.shape)

            x = np.zeros((len(genes), expression.shape[1]-1))
            for i, gene in enumerate(genes):
                mask = expression[:, 0] == gene
                if np.any(mask):
                    x[i] = expression[mask][0, 1:]
            X = np.concatenate([X, x], axis=1)


        if sublocalizations:
            path = os.path.join(DATA_ROOT, f'essential_genes/{organism}/SubLocalizations/subloc.csv')
            subloc = pd.read_csv(path).values
            print('Subcellular Localizations dataset shape:', subloc.shape)

            x = np.zeros((len(genes), subloc.shape[1]-1))
            for i, node in enumerate(genes):
                mask = subloc[:, 0] == node
                #print(subloc[:,0], node)
                if np.any(mask):
                    print(subloc[mask, 1:].astype(float).max(0))
                    x[i] = subloc[mask, 1:].astype(float).max(0)
            X = np.concatenate([X, np.array(x)], axis=1)

        # Cache ----------
        with open(cachepath, 'wb') as f:
            pickle.dump([edges, edge_weights, X, labels, genes], f, protocol=2)
        # ----------------

    train, test = train_test_split(labels, test_size=0.2, random_state=seed, stratify=labels[:, 1])

    print(f'Num nodes {len(genes)} ; num edges {len(edges)}')
    print(f'X.shape: {None if X is None else X.shape}.')
    print(f'Train labels. Num: {len(train)} ; Num pos: {train[:,1].sum()}')
    print(f'Test labels. Num: {len(test)} ; Num pos: {test[:,1].sum()}')
    return (edges, edge_weights), X, train, test, genes


if __name__ == '__main__':
    edge_info, X, train, test, genes = data(
                organism='melanogaster', 
                ppi='string', 
                expression=True,
                sublocalizations=False,
                orthologs=True)
