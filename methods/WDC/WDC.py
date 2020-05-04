import os
import numpy as np
import pandas as pd
import networkx as nx
from functools import reduce

import sys
sys.path.append('.')
from utils import yeast_data

def NIS(x):
    return (x - x.min()) / (x.max() - x.min())

def WDC(ppi, expression, labels, lambda_=.5):
    genes = reduce(np.intersect1d, [ppi[:, 0], ppi[:, 1], expression[:, 0]])
    N = len(genes)

    #mask = np.isin(expression[:, 0], genes)
    #expression = expression[mask]
    exp_genes = expression[:, 0]
    expression = expression[:, 1:]

    #mask1, mask2 = np.isin(ppi[:, 0], genes), np.isin(ppi[:, 1], genes)
    #ppi = ppi[mask1 & mask2]

    path = 'methods/WDC/ECC.npy'
    if not os.path.isfile(path):
        ECC = np.zeros((N, N))
        for (a, b) in ppi:
            mask_a = np.sum(ppi == a, axis=1, dtype=bool)
            neigs_a = np.unique(ppi[mask_a]) # plus self
            
            mask_b = np.sum(ppi == b, axis=1, dtype=bool)
            neigs_b = np.unique(ppi[mask_b])

            n_triangles = len(np.intersect1d(neigs_a, neigs_b)) - 2
            div = min(len(neigs_a), len(neigs_b)) -1
            ECC[genes == a, genes == b] = n_triangles / div
            print(ECC[genes ==a, genes==b])
        np.save(path, ECC)
    else:
        ECC = np.load(path)


    exp_genes = expression[:, 0]
    expression = expression[:, 1:].astype(np.float32)
    expre = np.zeros(N)
    for i, gene in enumerate(genes):
        expre[i] = expression[exp_genes == gene]

    PRR = np.corrcoef(expre)

    lambd = 0.5
    W = ECC * lambd + PRR * (1 - lambd)
    WDC = W.sum(1)

    return WDC


if __name__ == '__main__':
    root = '/home/schapke/projects/research/2020_FEB_JUN/data/'

    expression_path = '../data/yeast_final/Expression/Filtered.csv' 
    expression = pd.read_csv(expression_path).values

    biogrid_path = '../data/yeast_final/PPI/biogrid.csv'
    biogrid = pd.read_csv(biogrid_path).values

    print(biogrid.shape)
    mask = biogrid[:, 0] != biogrid[:, 1]
    biogrid = biogrid[mask]
    print(biogrid.shape)

    labels_path = '../data/yeast_final/EssentialGenes/ogee.csv'
    labels = pd.read_csv(labels_path).values

    _, train_labels, _ = yeast_data()

    solution = WDC(biogrid, expression, train_labels)
    solution_path = 'methods/WDC/results.npy'
    np.save(solution_path, solution)
