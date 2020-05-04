import sys; sys.path.append('.')

import os
import numpy as np
from tqdm import tqdm
import torch

from models.gat.gat_pytorch import GAT
from models.gat import params as gat_params
from utils import *
from runners import tools

import lime
import lime.lime_tabular


# Args ----------------------------------------------------------
organism = 'human'
ppi = 'string'
expression = True
sublocs = False
orthologs = False

ROOT = '/home/schapke/projects/research/2020_FEB_JUN/src'
weightsdir = os.path.join(ROOT, 'models/gat/weights') 
snapshot_name = f'{organism}_{ppi}_expression'
savepath = os.path.join(weightsdir, snapshot_name)

GENE = 'Q6NW34'
FEATURES = [f'expression_{i}' for i in range(64)] + ['degree']
# ---------------------------------------------------------------

def get_neighbors(edges, genes, gene):
    idx = np.where(genes == gene)[0][0]
    x, y = torch.where(edges == idx)
    return np.unique(edges[1-x, y])

def choose_gene(model, X, edges, genes, indexes, labels=None):
    model.eval()
    with torch.no_grad():
        outs = model(X, edges)

    probs = torch.sigmoid(outs).squeeze().numpy()
    idxs = probs.argsort()[::-1]

    for idx in idxs:
        if idx in indexes:
            i = indexes.index(idx)
            x, y = torch.where(edges == idx)
            neigs = edges[1-x, y]

            if labels is not None:
                print(probs[idx], labels[i], len(neigs))
                if labels[i] == 1 and probs[idx] > 0.5 and 2 < len(neigs) < 10:
                    break
            else:
                print(probs[idx], len(neigs))
                break

    gene = genes[idx] 
    return gene, idx


class Wrapper:
    def __init__(self, model, X, edges, indexes, node_index, shape):
        model.eval()
        self.model = model
        self.X = X
        self.edges = edges
        self.indexes = indexes
        self.node_index = node_index
        self.shape = shape

    def __call__(self, x):
        X = self.X.clone() 
        x = torch.tensor(x, dtype=torch.float32)

        out = np.zeros((len(x), 2))
        out[:, 1] = 1

        print('x.shape:', x.shape)
        for i in tqdm(range(len(x))):
            X[self.indexes] = x[i].reshape((*self.shape))

            with torch.no_grad():
                y = self.model(X, self.edges)
            node_y = y[self.node_index]
            node_y = torch.sigmoid(node_y).numpy()
            out[:, 0] = node_y
            out[:, 1] -= node_y

        return out


def main(seed=0):
    set_seed(seed)

    # Getting the data ----------------------------------
    params = {
        'organism': organism,
        'ppi': ppi,
        'expression': expression,
        'sublocs': sublocs,
        'orthologs': orthologs,
        'string_thr': 700,
        'no_ppi': False,
        'use_weights': False,
    }

    (edges, _), X, (train_idx, train_y), (val_idx, val_y), \
            (test_idx, test_y), genes = tools.get_data(params, seed=0)
    print('Fetched data')
    # ---------------------------------------------------


    # Model ---------------------------------------------
    snapshot = torch.load(savepath)
    model = GAT(in_feats=X.shape[1], **snapshot['model_params'])
    model.load_state_dict(snapshot['model_state_dict'])
    print('Model loaded. Val AUC: {}'.format(snapshot['auc']))
    # ---------------------------------------------------

    GENE, gene_idx = choose_gene(model, X, edges, genes, test_idx, test_y)

    #gene_idx = np.where(genes == GENE)[0][0]

    neigs_idx = get_neighbors(edges, genes, GENE)
    neigs = [genes[idx] for idx in neigs_idx]

    def label(p):
        return 'Essential' if p > 0.5 else 'Non-Essential'

    print(f'Chosen Gene: {GENE}')
    print(f'Label: {label(test_y[test_idx.index(gene_idx)])}')
    print(f'Prediction: {label(1)}')
    print(f'Neighbors: {neigs}')


    # Select Node
    indexes = np.union1d(neigs_idx, gene_idx)
    names = np.union1d([GENE], neigs)

    X_modif = X.clone()[indexes]
    shape = X_modif.shape
    predict_fn = Wrapper(model, X, edges, indexes, gene_idx, shape)
    X_modif = X_modif.reshape((-1)).numpy()

    feats = X.shape[1]
    X_train = np.zeros((len(X), len(X_modif)))
    for n, neig in enumerate(neigs):
        X_train[:, n*feats:(n+1)*feats] = X[:, :]

    fnames = []
    for name in names:
        fnames += [name + '_' + f for f in FEATURES]

    print(fnames)
    print(X_train.shape)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=fnames, 
        class_names=['Non-Essential', 'Essential'], discretize_continuous=True)

    exp = explainer.explain_instance(X_modif, predict_fn, num_features=20)

    _expression = 0
    _expression_abs = 0
    _degree = 0
    _degree_abs = 0
    for o in exp.as_list():
        if 'expression' in o[0]:
            _expression += o[1]
            _expression_abs += abs(o[1])
        elif 'degree' in o[0]:
            _degree += o[1]
            _degree_abs += abs(o[1])
        print(o)

    print(f'Expression: {_expression} / {_expression_abs}')
    print(f'Degree: {_degree} / {_degree_abs}')



if __name__ == '__main__':
    main()


