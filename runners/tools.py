import sys; sys.path.append('.')
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split as tts
import torch.nn.functional as F

from utils import *

class Loss():
    def __init__(self, y, idx):
        self.y = y
        idx = np.array(idx)

        self.y_pos = y[y == 1]
        self.y_neg = y[y == 0]

        self.pos = idx[y.cpu() == 1]
        self.neg = idx[y.cpu() == 0]


    def __call__(self, out):
        loss_p = F.binary_cross_entropy_with_logits(out[self.pos].squeeze(), self.y_pos)
        loss_n = F.binary_cross_entropy_with_logits(out[self.neg].squeeze(), self.y_neg)
        loss = loss_p + loss_n
        return loss


class WeightAveraging:
    def __init__(self, model, start, rate):
        self.model = model
        self.iter = 0
        self.start = start
        self.rate = rate

        self.weights = [list(model.parameters())]

    def step(self):
        self.iter += 1
        if self.iter < self.start and \
                (self.iter % self.rate) == 0:
            self.weights.append(list(self.model.parameters()))

    def set_weights(self):
            weights = []
            for i in range(len(self.weights[0])):
                pars = [weight[i] for weight in self.weights]
                weight = torch.stack(pars, dim=0)
                weight = weight.mean(0)
                weights.append(weight)

            params = self.model.parameters()
            for old, new in zip(params, weights):
                old.data = new

def get_args():
    # Args ----------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--n_runs', type=int, default=0)
    parser.add_argument('--organism', default='yeast')
    parser.add_argument('--ppi', default='biogrid')
    parser.add_argument('--string_thr', default=500, type=int, help='Connection threshold for STRING PPI database')
    parser.add_argument('--expression', action='store_true')
    parser.add_argument('--orthologs', action='store_true')
    parser.add_argument('--sublocs', action='store_true')
    parser.add_argument('--no_ppi', action='store_true', help='Run GNN without the interaction network')
    parser.add_argument('--use_weights', action='store_true', help='Wether to use StringDB weights for connections')
    parser.add_argument('--name', default='', help="Name for the results csv")
    parser.add_argument('--weightsdir', default='', help="Directory for the model's weights")
    parser.add_argument('--outdir', default='')
    args = parser.parse_args()
    return args


def get_data(args, seed=0, parse=True, weights=False):
    # Getting the data ----------------------------------

    (edges, edge_weights), X, train_ds, test_ds, genes = data(
            organism=args['organism'], 
            ppi=args['ppi'], 
            expression=args['expression'], 
            orthologs=args['orthologs'],
            sublocalizations=args['sublocs'],
            seed=seed,
            string_thr=args['string_thr'],
            weights=weights)

    if edge_weights is None:
        edge_weights = np.ones(len(edges))

    N = len(X)
    mapping = dict(zip(genes, range(N)))
    # ---------------------------------------------------

    # Preprocessing -------------------------------------
    # Remove self loops
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]
    edge_weights = edge_weights[mask]

    # Removes repeated connections
    df = pd.DataFrame({'A': edges[:, 0], 'B': edges[:, 1]})
    df = df.drop_duplicates()
    edges = df.values
    indexes = df.index.values
    edge_weights = edge_weights[indexes]
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

    edge_index = np.vectorize(mapping.__getitem__)(edges)
    if args['no_ppi']:
        edges = np.ones((N, 2), dtype=np.int)
        edges[:, 0] = range(N)
        edges[:, 1] = range(N)

    if not args['use_weights']:
        edge_weights = None

    degrees = np.zeros((N, 1))
    nodes, counts = np.unique(edge_index, return_counts=True)
    degrees[nodes, 0] = counts

    edge_index = torch.from_numpy(edge_index.T)
    edge_index = edge_index.to(torch.long).contiguous()

    if X is None or not X.shape[1]:
        X = np.random.random((N, 50))

    X = np.concatenate([X, degrees.reshape((-1, 1))], 1)
    X = torch.from_numpy(X).to(torch.float32)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
        
    train, val = tts(train_ds, test_size=0.2, stratify=train_ds[:, 1], random_state=seed)

    train_idx = [mapping[t] for t in train[:, 0]]
    train_y = torch.tensor(train[:, 1].astype(int), dtype=torch.float32)

    val_idx = [mapping[v] for v in val[:, 0]]
    val_y = torch.tensor(val[:, 1].astype(int), dtype=torch.float32)

    test_idx = [mapping[v] for v in test_ds[:, 0]]
    test_y = torch.tensor(test_ds[:, 1].astype(int), dtype=torch.float32)
    # ---------------------------------------------------

    print(f'\nNumber of edges in graph: {len(edges)}')
    print(f'Number of features: {X.shape[1]}')
    print(f'Number of nodes in graph: {len(X)}\n')
    print('Using Edge Weights' if edge_weights is not None else 'Not using edge weights')

    if not parse:
        return (edge_index, edge_weights), X, train_ds, test_ds, genes

    return (edge_index, edge_weights), X, (train_idx, train_y), (val_idx, val_y), (test_idx, test_y), genes
