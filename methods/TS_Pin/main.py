'''
TSPIN algorithm.
Paper: https://www.sciencedirect.com/science/article/pii/S0022519318301437
'''

import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
import optuna

import sys
sys.path.append('.')
from utils import yeast_data

spp_order = [
    'Nucleus',
    'Mitochondrion',
    'Cytosol',
    'Endoplasmic reticulum',
    'Cytoskeleton',
    'Plasma membrane',
    'Golgi apparatus',
    'Endosome',
    'Peroxisome',
    'Extracellular space',
    'Lysosome',
    ]


def main(sublocations, edge_list, expression, labels):
    express_genes = expression[:, 0]
    express = expression[:, 1:].astype(float)

    mu = express.mean(axis=1)
    sigma = express.std(axis=1)
    k = 2.5
    F = 1 / (1 + sigma ** 2)

    active_th = mu + k * sigma * (1 - F)
#   the gene products
#   with an expression value lower than 0.7 are filtered.





if __name__ == '__main__':
    sublocations_path = '../data/yeast_final/SubLocalizations/Localizations11.csv' 
    sublocations = pd.read_csv(sublocations_path)

    biogrid_path = '../data/yeast_final/PPI/biogrid.csv'
    biogrid = pd.read_csv(biogrid_path)

    expression_path = '../data/yeast_final/Expression/Filtered.csv' 
    expression = pd.read_csv(expression_path).values

    labels_path = '../data/yeast_final/EssentialGenes/ogee.csv'
    labels = pd.read_csv(labels_path)

    _, train_labels, _ = yeast_data()

    main(sublocations, biogrid, expression, train_labels)
