
import os
import sys; sys.path.append('.')
import pandas as pd
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from models.gat.gat_pytorch import GAT
from runners.tools import get_data
from torch_geometric.utils import remove_isolated_nodes, remove_self_loops

# Args ----------------------------------------------------------
organism = 'melanogaster'
ppi = 'string'
expression = True
sublocs = False
orthologs = False

weightsdir = './models/gat/weights'
snapshot_name = f'{organism}_{ppi}'
snapshot_name += '_expression' if expression else ''
snapshot_name += '_orthologs' if orthologs else ''
snapshot_name += '_sublocs' if sublocs else ''
savepath = os.path.join(weightsdir, snapshot_name)
# ---------------------------------------------------------------

def single_node():
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


def multi_node():
    pass

def save_attention(edge_index, att, labels, genes):
    # ------------------- Saving attention network ----------------------------------- #    
    att = att.max(1).reshape((-1, 1))
    print(edge_index.shape, labels.shape, att.shape)
    t = lambda t: torch.tensor(t)
    edge_index, att = remove_self_loops(t(edge_index.T), t(att))
    edge_index, att, _ = remove_isolated_nodes(edge_index, att)
    edge_index = edge_index.numpy().T
    att = att.numpy()
    print(edge_index.shape, labels.shape, att.shape)
    nodes_edges = np.unique(edge_index.reshape((-1)))
    nodes_idx = np.intersect1d(np.arange(len(labels)), nodes_edges)

    labels = labels[nodes_idx]
    genes = genes[nodes_idx]

    meta_edges = pd.DataFrame(np.concatenate([edge_index.astype(int), att], 1))
    meta_edges.to_csv(f'../data/essential_genes/gat_attention/{organism}_edges.csv', index=False)
    
    meta_nodes = pd.DataFrame(np.stack([nodes_idx, genes, labels], 1))
    path = f'../data/essential_genes/gat_attention/{organism}_nodes.csv'
    meta_nodes.to_csv(path, index=False)
    print(meta_edges.shape, meta_nodes.shape)
    print('Saved edges and nodes to ', path)
    # -------------------------------------------------------------------------------- #    


def main():

    update = True
    cache = '.cache/int_cache.npy'
    os.makedirs('./cache', exist_ok=True)

    if os.path.isfile(cache) and not update:
        outs, att, edge_index, genes, train_idx, test_idx, val_idx, test_y, train_y, val_y = np.load(cache, allow_pickle=True)
    else:
        # Getting the data ----------------------------------
        params = {
            'organism': organism,
            'ppi': ppi,
            'expression': expression,
            'sublocs': sublocs,
            'orthologs': orthologs,
            'no_ppi': False,
            'use_weights': False,
            'string_thr': 500
        }
        (edges, _), X, (train_idx, train_y), (val_idx, val_y), \
                (test_idx, test_y), genes = get_data(params, seed=0)

        print('Fetched data')
        # ---------------------------------------------------


        # Model ---------------------------------------------
        snapshot = torch.load(savepath)
        model = GAT(in_feats=X.shape[1], **snapshot['model_params']).cpu()
        model.load_state_dict(snapshot['model_state_dict'])
        model.cuda().eval()
        print('Model loaded. Val Auc: {}'.format(snapshot['auc']))

        with torch.no_grad():
            outs, alphas, edge_index = model(X.cuda(), edges.cuda(), return_alphas=True) 

        outs = torch.sigmoid(outs).cpu().numpy().squeeze()
        att = torch.cat(alphas, dim=1)
        att = att.cpu().numpy()
        edge_index = edge_index.cpu().numpy().T
        genes = np.array(genes)
        # ---------------------------------------------------

        np.save(cache, [outs, att, edge_index, genes, train_idx, test_idx, val_idx, test_y, train_y, val_y])

    labels = np.zeros(edge_index.max()) -1
    labels[train_idx] = train_y
    labels[test_idx] = test_y + 2
    labels[val_idx] = val_y + 4
    save_attention(edge_index, att, labels, genes)
    return


    G = nx.DiGraph()
    G.add_edges_from(edge_index)
    

    #for node in list(G.nodes):
    #    print(node, len(list(G.neighbors(node))))
    #return


    nodes = [2412]
    neigs = list(G.neighbors(nodes[0]))
    nodes += neigs

    for neig in neigs:
        if len(nodes) > 500:
            break

        n = list(G.neighbors(neig))
        nodes += n
        nodes = np.unique(nodes).tolist()

    all_nodes = list(G.nodes())
    diff = np.setdiff1d(all_nodes, nodes)
    G.remove_nodes_from(diff)
    G.remove_edges_from(list(G.edges()))

    mask1 = np.isin(edge_index[:, 0], nodes)
    mask2 = np.isin(edge_index[:, 1], nodes)
    mask = (mask1 & mask2)
    edge_index = edge_index[mask]

    att = att[mask]
    att = att.mean(1)
    mask = att > 0.03

    att = att[mask]
    edge_index = edge_index[mask]

    att = att / att.max()
    att[att < 0.10] = .10

    color = np.zeros((len(att), 4))
    color[:, 2] = 0.35 
    color[:, 1] = 0.4 
    color[:, 0] = 1 
    color[:, 3] = att

    G.add_edges_from(edge_index)
    G.remove_nodes_from(list(nx.isolates(G)))
    nodes = list(G.nodes)

    node_colors = np.zeros((len(G.nodes), 4)) 
    node_colors[:, 3] = 0.8
    for node in nodes:
        y = -1
        if node in train_idx:
            idx = train_idx.index(node)
            y = train_y[idx]

        elif node in test_idx:
            idx = test_idx.index(node)
            y = test_y[idx]
            i = nodes.index(node)

        elif node in val_idx:
            idx = val_idx.index(node)
            y = val_y[idx]


        node = nodes.index(node)
        if y == 1:
            node_colors[node, 1] = 1
        elif y == 0:
            node_colors[node, 0] = 1
        else:
            node_colors[node, 3] = 0.5

    print('Len nodes', len(G.nodes))
    print('Num Edges', len(G.edges))
    print('Att:', att.min(), att.mean(), att.max())

    plt.figure(figsize=(8, 6))
    nx.draw_kamada_kawai(
            G, 
            arrows=True, 
            node_size=20, 
            node_color=node_colors, 
            edge_color=color, 
            width=1
            ) 
    plt.savefig('attention.png')
    plt.show()
    # ---------------------------------------------------


if __name__ == '__main__':
    main()


