import os
import numpy as np
import pandas as pd
import networkx as nx
import json
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from functools import reduce

import sys
sys.path.append('.')
from utils import yeast_data


def NIS(x):
    return (x - x.min()) / (x.max() - x.min())

def SCP(ppi, sublocations, expression, labels, lambda_=.5):
    C, N = np.unique(sublocations[:, 1], return_counts=True)
    print(C)
    print(N)

    ISC = 1 / N
    isc_mapping = dict(zip(C, ISC))

    intersection = reduce(np.intersect1d, [ppi[:, 0], ppi[:, 1], sublocations[:, 0], expression[:, -1]])

    n = len(intersection)
    mapping = dict(zip(intersection, np.arange(n)))

    # Setting Weights
    W = np.zeros((n, n))
    
    for a, b in ppi:
        if a not in intersection or b not in intersection:
            continue

        mask = sublocations[:, 0] == a
        subA = np.unique(sublocations[mask, 1])

        mask = sublocations[:, 0] == b
        subB = np.unique(sublocations[mask, 1])

        inter = np.intersect1d(subA, subB)
        if len(inter) == 0:
            union = np.union1d(subA, subB) 
            iscs = [isc_mapping[c] for c in union]
            weight = min(iscs)
        else:
            iscs = [isc_mapping[c] for c in inter]
            weight = max(iscs)

        a, b = mapping[a], mapping[b]
        W[a, b] = weight
        W[b, a] = weight

    IPSC = W.sum(0)

    MPR = np.zeros(n)
    for gene, val in labels:
        if gene in mapping:
            MPR[mapping[gene]] = val

    for i in range(20):
        alpha = 0.85 
        M_hat_one = W / W.sum(0)
        M_hat_one[np.isnan(M_hat_one)] = 0

        M_hat_two = IPSC / IPSC.sum() 
        M_hat = alpha * M_hat_one + (1 - alpha) * M_hat_two

        MPR_tilde = M_hat @ MPR

        MPR = MPR_tilde / np.linalg.norm(MPR_tilde, ord=2)
    

    express = np.zeros((n, expression.shape[1]-1))
    genes = expression[:, -1]
    for i, gene in enumerate(genes):
        if gene in intersection:
            express[mapping[gene]] = expression[i, :-1]

    PCC = np.corrcoef(express)
    W[W != 0] = 1
    IPCC = (W * PCC).sum(1)

    scores = lambda_ * NIS(MPR) + (1 - lambda_) * NIS(IPCC)

    results = np.stack([intersection, scores], axis=1)
    return results


def plot_results(solution, essential, non_essential):
    preds = []
    y = []
    y_names = []

    for gene in essential:
        mask = solution.values[:, 0] == gene
        score = solution[mask]['score'].values
        if len(score):
            preds.append(score[0])
            y_names.append(gene)
            y.append(1)

    for gene in non_essential:
        mask = solution.values[:, 0] == gene
        score = solution[mask]['score'].values
        if len(score):
            preds.append(score[0])
            y_names.append(gene)
            y.append(0)

    gt_path = 'EssentialGenes/ground_truth.csv'
    if not os.path.isfile(gt_path):
        gt = pd.DataFrame({'gene_name': y_names, 'label': y})
        gt.to_csv(gt_path, index=False)

    fpr, tpr, _ = roc_curve(y, preds) 
    roc_auc = roc_auc_score(y, preds)
    print('AUC SCORE:', roc_auc)

    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=f'SCP (area = %0.2f)' % roc_auc)
    
    # Add DC predictions to the plot for comparison
    dc_path = 'DC_preds.npy'
    if os.path.isfile(dc_path):
        dc_preds = np.load(dc_path)
        fpr, tpr, _ = roc_curve(y, dc_preds) 
        roc_auc = roc_auc_score(y, dc_preds)
        plt.plot(fpr, tpr, lw=lw, label=f'DC (area = %0.2f)' % roc_auc)


    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Results on Essential Genes dataset')
    plt.legend(loc="lower right")
    #plt.savefig('results.png')
    #plt.show()




if __name__ == '__main__':
    root = '/home/schapke/projects/research/2020_FEB_JUN/data/'

    sublocations_path = '../data/yeast_final/SubLocalizations/Localizations11.csv' 
    sublocations = pd.read_csv(sublocations_path).values

    expression_path = '../data/yeast_final/Expression/Filtered.csv' 
    expression = pd.read_csv(expression_path).values

    biogrid_path = '../data/yeast_final/PPI/biogrid.csv'
    biogrid = pd.read_csv(biogrid_path).values

    labels_path = '../data/yeast_final/EssentialGenes/ogee.csv'
    labels = pd.read_csv(labels_path).values

    _, train_labels, _ = yeast_data()

    solution = SCP(biogrid, sublocations, expression, train_labels)
    solution_path = 'methods/SCP/results.npy'
    np.save(solution_path, solution)


def old():
    root = '/home/schapke/projects/research/2020_FEB_JUN/data/SCP/'

    essential_df = pd.read_csv(os.path.join(root, 'EssentialGenes', 'DEG_yeast_essential.csv'))
    non_essential_df = pd.read_csv(os.path.join(root, 'EssentialGenes', 'DEG_yeast_non_essential.csv'))

    expression_df = pd.read_csv(os.path.join(root, 'GeneExpression', 'profiles.txt'), delimiter='\t')
    expression = expression_df.values[:, 1:-1].astype(np.float32)

    sublocations_df = pd.read_csv(os.path.join(root, 
        'SubcellularLocalization', 'yeast_compartment_knowledge_full.tsv'), delimiter='\t', header=None)

     
    print('\n--- Essential Genes ---')
    essential_genes = essential_df['Gene_Ref'].unique().tolist()
    print(len(essential_genes))

    print('\n--- Non Essential Genes ---')
    non_essential_genes = non_essential_df['Gene_Ref'].unique().tolist()
    print(len(non_essential_genes))

    print('\n--- Expression Genes ---')
    expression_genes = expression_df['Gene'].unique().tolist()
    print(len(expression_genes))

    print('\n--- SubLocation Genes ---')
    gene_a = sublocations_df[1].unique()
    gene_b = sublocations_df[0].unique()
    print(len(np.unique(gene_a)))

    def IOU(arr1, arr2):
        #return len(np.intersect1d(arr1, arr2)) / len(np.union1d(arr1, arr2))
        return f'Intersection: {len(np.intersect1d(arr1, arr2))} | Max Intersection: {min(len(arr1), len(arr2))}'

    essential_path = 'EssentialGenes/essential_genes_mod.npy'
    if not os.path.isfile(essential_path):
        gene_a = [a.lower() for a in gene_a]
        mapping = dict(zip(gene_a, gene_b))
        essential_genes = [mapping[g.lower()] for g in essential_genes if g.lower() in mapping.keys()]
        np.save(essential_path, essential_genes)
    else:
        essential_genes = np.load(essential_path)


    non_essential_path = 'EssentialGenes/non_essential_genes_mod.npy'
    if not os.path.isfile(non_essential_path):
        gene_a = [a.lower() for a in gene_a]
        mapping = dict(zip(gene_a, gene_b))
        non_essential_genes = [mapping[g.lower()] for g in non_essential_genes if g.lower() in mapping.keys()]
        np.save(non_essential_path, non_essential_genes)
    else:
        non_essential_genes = np.load(non_essential_path)

    network_path = 'PPI/network.pickle'
    if not os.path.isfile(network_path):
        G = nx.Graph()
        G.add_edges_from(zip(A, B))
        G = nx.convert_node_labels_to_integers(G, label_attribute='names')
        for node, data in G.nodes(data=True):
            names = np.array(eval(data['names']))
            G.nodes[node]['names'] = names
            mask = np.array([np.any(gene_a == n) for n in names])
            if not np.any(mask):
                continue
            name = names[mask][0]
            rows = sublocations_df[sublocations_df[1] == name]
            sublocations = rows[3].unique()
            G.nodes[node]['sublocations'] = sublocations

        G.remove_edges_from(nx.selfloop_edges(G))
        nx.write_gpickle(G, network_path)
    else:
        G = nx.read_gpickle(network_path)

    network_path = 'PPI/network_filtered.pickle'
    if not os.path.isfile(network_path):
        for node, info in list(G.nodes(data=True)):
            names = info['names']
            for name in names:
                if name in expression_genes:
                    G.nodes[node]['name'] = name
                    break
            if 'name' not in G.nodes[node]:
                G.remove_node(node)
        G = nx.convert_node_labels_to_integers(G)
        nx.write_gpickle(G, network_path)
    else:
        G = nx.read_gpickle(network_path)

    name_map = nx.get_node_attributes(G, 'name')
    names = list(name_map.values())

    expression_path = 'GeneExpression/expression_filtered.npy'
    if not os.path.isfile(expression_path):
        expression = []
        expression_names = []
        for row in expression_df.values:
            name = row[0]
            vals = row[1:-1]
            if name not in expression_names and \
                    name in names:
                expression.append(vals)
                expression_names.append(name)
        expression = np.array(expression)
        expression_names = np.array(expression_names).reshape((-1, 1))
        expression = np.concatenate([expression_names, expression], axis=1)
        np.save(expression_path, expression)
    else:
        expression = np.load(expression_path, allow_pickle=True)

    print(f'\nExpression and Essential: {IOU(essential_genes, expression[:, 0])}')
    print(f'Expression and Non-Essential: {IOU(non_essential_genes, expression[:, 0])}')
    print(f'Essential and SubLocs type 2: {IOU(essential_genes, gene_b)}')
    print(f'Non-Essential and SubLocs type 2: {IOU(non_essential_genes, gene_b)}')
    print(f'Expression and SubLocs type 2: {IOU(expression[:, 0], gene_b)}')

    solution_path = 'solution.csv'
    update = True
    if not os.path.isfile(solution_path) or update:
        solution = SCP(G)
        solution.to_csv(solution_path, index=False)
    else:
        solution = pd.read_csv(solution_path)
    print(solution.head())

    plot_results(solution, essential_genes, non_essential_genes)
