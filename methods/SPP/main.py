'''
SPP algorithm.
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

RANKS = [0.7016382196334383, 0.025385463253632525, 0.008546478647931878, 0.008357987341182527, 0.006135859434980569, 0.004395003625837491, 0.0023066059605159317, 0.0008972164411409271, 0.0006487468197490792, 0.0006301450312384, 0.00021056874178155906]

def spp(ppi, ranks=RANKS):
    nodes = np.array(ppi.nodes())
    n = len(nodes)
    SCN = np.zeros((n, n))
    SPP = np.zeros(n)

    #for i, node in enumerate(nodes):
    #    deg_node = ppi.degree[node]
    #    neigs = list(ppi.neighbors(node))
    #    for neig in neigs:
    #        j = np.any(nodes == neig)
    #        neig_neigs = list(ppi.neighbors(neig))
    #        deg_neig = ppi.degree[neig]

    #        scn = len(np.intersect1d(neigs, neig_neigs))
    #        SCN[i, j] = scn

    #        frac = max(deg_node, deg_neig) / (deg_neig + deg_node)
    #        SPP[i] += frac * scn

    path = 'methods/SPP/tmp.npy'
    #np.save(path, SPP)
    SPP = np.load(path)

    sublocs = nx.get_node_attributes(ppi, 'subloc')
    keys = np.array(list(sublocs.keys()))
    values = np.array(list(sublocs.values()))
    for i, subloc in enumerate(spp_order):
        #rank = len(spp_order) - i
        rank = ranks[i]
        mask = values == subloc

        sel_nodes = keys[mask]
        mask = np.isin(nodes, sel_nodes)

        vals = SPP[mask]
        vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        SPP[mask] = vals + rank
    
    results = np.stack([nodes, SPP], axis=1)
    return results



def main(sublocations, edge_list, labels):
    ppi = nx.Graph()
    ppi.add_edges_from(edge_list.values)

    ppi_genes = ppi.nodes()
    subloc_genes = sublocations['0']
    inter = np.intersect1d(ppi_genes, subloc_genes)

    path = 'methods/SPP/temp_g.gpickle'
    if not os.path.isfile(path):
        for node in list(ppi_genes):
            if node not in inter:
                ppi.remove_node(node)
            else:
                mask = subloc_genes == node
                sublocs = sublocations[mask]['3'].unique()
                for sl in spp_order:
                    if sl in sublocs:
                        ppi.nodes[node]['subloc'] = sl
                        break


        nodes = list(ppi.nodes())
        n = len(nodes)

        # remove connections from proteins in different sublocations
        for node in nodes:
            n_subloc = ppi.nodes[node]['subloc']
            for neig in list(ppi.neighbors(node)):
                neig_subloc = ppi.nodes[neig]['subloc']
                if n_subloc != neig_subloc:
                    ppi.remove_edge(node, neig)

        nx.write_gpickle(ppi, path)
    else:
        ppi = nx.read_gpickle(path)

    path = 'methods/SPP/predictions_yeast.npy'
    predictions = spp(ppi)
    np.save(path, predictions)
    #predictions = np.load(path)

    genes = labels[:, 0]
    mask = np.isin(predictions[:, 0], genes)
    predictions = predictions[mask]

    pred_ordered, y_ordered = [], []
    for (gene, val) in predictions:
        pred_ordered.append(float(val))
        mask = genes == gene
        val = labels[mask, 1]
        y_ordered.append(float(val))

    roc_auc = roc_auc_score(y_ordered, pred_ordered)
    print('Roc Auc:', roc_auc)
    return roc_auc




if __name__ == '__main__':
    sublocations_path = '../data/yeast_final/SubLocalizations/Localizations11.csv' 
    sublocations = pd.read_csv(sublocations_path)

    biogrid_path = '../data/yeast_final/PPI/biogrid.csv'
    biogrid = pd.read_csv(biogrid_path)

    labels_path = '../data/yeast_final/EssentialGenes/ogee.csv'
    labels = pd.read_csv(labels_path)

    _, train_labels, _ = yeast_data()

    main(sublocations, biogrid, train_labels)
