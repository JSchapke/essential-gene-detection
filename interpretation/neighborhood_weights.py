
import os
import sys; sys.path.append('.')
import pandas as pd
import numpy as np
import torch
import pickle
from models.gat.gat_pytorch import GAT
from runners.tools import get_data

# Args ----------------------------------------------------------
organism = 'human'
ppi = 'dip'
expression = True
sublocs = True
orthologs = False

ROOT = '/home/schapke/projects/research/2020_FEB_JUN/src'
weightsdir = os.path.join(ROOT, 'models/gat/weights') 
snapshot_name = f'{organism}_{ppi}_expression_sublocs'
savepath = os.path.join(weightsdir, snapshot_name)
# ---------------------------------------------------------------

def main():
    # Getting the data ----------------------------------
    cache = 'interpretation/cache.pickle'
    params = {
        'organism': organism,
        'ppi': ppi,
        'expression': expression,
        'sublocs': sublocs,
        'orthologs': orthologs,
        'no_ppi': False
    }
    #(edges, _), X, (train, test, genes = get_data(params, seed=0, parse=False)
    (edges, _), X, (train_idx, train_y), (val_idx, val_y), \
            (test_idx, test_y), genes = get_data(params, seed=0)

    print('Fetched data')
    # ---------------------------------------------------


    # Model ---------------------------------------------
    cache = './.cache/interpretation_results.pth'
    if not os.path.isfile(cache):
        snapshot = torch.load(savepath)
        model = GAT(in_feats=X.shape[1], **snapshot['model_params']).cpu()
        model.load_state_dict(snapshot['model_state_dict'])
        model.cpu().eval()
        print('Model loaded. Val AUC: {}'.format(snapshot['val_AUC']))

        with torch.no_grad():
            outs, alphas, _ = model(X, edges, return_alphas=True) 

        torch.save([outs, alphas], cache)
    else:
        outs, alphas = torch.load(cache)

    outs = torch.sigmoid(outs).numpy().squeeze()
    # ---------------------------------------------------

    # Processing ----------------------------------------
    idx = outs.argsort()[::-1]
    idx = [i for i in idx if i in test_idx]
    for i in idx:
        j = test_idx.index(i)
        if test_y[j] == 1:
            break

    gene_idx = i
    gene = genes[i]
    prob = outs[i]
    print(gene, prob)

    gene_idx = np.where(genes == gene)[0][0]
    edges = edges.numpy()
    y, x = np.where(edges == gene_idx)
    neigs = edges[y, 1-x]
    print('Neighbors:', neigs)


    print('Train Nodes:')

    nodes = [[gene, 'prediction', prob, -1]]

    for i, neig_idx in enumerate(neigs):
        alpha = (alphas[0][y[i]] + alphas[1][y[i]]) / 2
        if neig_idx in test_idx:
            idx = test_idx.index(neig_idx)
            print('Test node:', genes[neig_idx], test_y[idx])
            nodes.append([genes[neig_idx], 'test', test_y[idx], alpha])
        elif neig_idx in train_idx:
            idx = train_idx.index(neig_idx)
            print('Train node:', genes[neig_idx], train_y[idx])
            nodes.append([genes[neig_idx], 'train', train_y[idx], alpha])
        elif neig_idx in val_idx:
            idx = val_idx.index(neig_idx)
            print('Validation node:', genes[neig_idx], val_y[idx])
            nodes.append([genes[neig_idx], 'val', val_y[idx], alpha])
        else:
            print('Unkown node:', genes[neig_idx])
            nodes.append([genes[neig_idx], 'unkwown', -1, alpha])

    print(alphas[0].shape, edges.shape)
    print(len(genes))
    print(len(edges) + len(genes))

    print(nodes)


    # ---------------------------------------------------


if __name__ == '__main__':
    main()


