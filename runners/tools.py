from utils import utils
import torch.nn.functional as F
from sklearn.model_selection import train_test_split as tts
import torch
import pandas as pd
import numpy as np
import argparse
import sys
sys.path.append('.')


class Loss():
    def __init__(self, y, idx):
        self.y = y
        idx = np.array(idx)

        self.y_pos = y[y == 1]
        self.y_neg = y[y == 0]

        self.pos = idx[y.cpu() == 1]
        self.neg = idx[y.cpu() == 0]

    def __call__(self, out):
        loss_p = F.binary_cross_entropy_with_logits(
            out[self.pos].squeeze(), self.y_pos)
        loss_n = F.binary_cross_entropy_with_logits(
            out[self.neg].squeeze(), self.y_neg)
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


def get_args(parse=True):
    # Args ----------------------------------------------------------
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', action='store_true',
                        help='Train the model (else will search for saved checkpoints)')
    parser.add_argument('--hyper_search', action='store_true',
                        help='Hyperparamters search')
    parser.add_argument('--n_runs', type=int, default=0,
                        help='How many runs to perform for consistency')
    parser.add_argument('--organism', default='yeast',
                        help='Organism. ["yeast", "coli", "melanogaster", "human"]')
    parser.add_argument('--ppi', default='biogrid',
                        help='PPI Network. ["biogrid", "string", "dip"]')
    parser.add_argument('--string_thr', default=500, type=int,
                        help='Connection threshold for STRING PPI database')
    parser.add_argument('--expression', action='store_true',
                        help='Wheter to use expression data')
    parser.add_argument('--orthologs', action='store_true',
                        help='Wheter to use orthology data')
    parser.add_argument('--sublocs', action='store_true',
                        help='Wheter to use subcelulr localization data')
    parser.add_argument('--no_ppi', action='store_true',
                        help='Run GNN without the interaction network')
    parser.add_argument('--use_weights', action='store_true',
                        help='Wether to use StringDB weights for connections')
    parser.add_argument('--name', default='', help="Name for the results csv")
    parser.add_argument('--weightsdir', default='',
                        help="Directory for the model's weights")
    parser.add_argument('--seed', default=0, type=int,
                        help="Seed used for training")
    parser.add_argument('--outdir', default='', help='Output directory')

    if not parse:
        return parser

    args = parser.parse_args()
    return args


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


def get_data(args, seed=0, parse=True, weights=False):
    # Getting the data ----------------------------------
    default_args = dict(string_thr=500, use_weights=False, no_ppi=False)
    args = dict(default_args, **args)

    (edges, edge_weights), X, train_ds, test_ds, genes = utils.data(
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

    if X is None or not X.shape[1]:
        X = np.random.random((N, 50))

    if X.shape[1] < 50:
        X = np.concatenate([X, np.random.random((N, 50))], axis=1)

    X = np.concatenate([X, degrees.reshape((-1, 1))], 1)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)

    train, val = tts(train_ds, test_size=0.05,
                     stratify=train_ds, random_state=seed)

    train_idx = [mapping[t] for t in train.index]
    val_idx = [mapping[v] for v in val.index]
    test_idx = [mapping[v] for v in test_ds.index]

    # Feature selection -------------------------------------
    k = 300
    if args['organism'] == 'coli':
        k = 50
    elif args['organism'] == 'human':
        k = 150
    elif args['organism'] == 'yeast':
        k = 120
    elif args['organism'] == 'melanogaster':
        k = 100

    red_idx = np.concatenate([train_idx, test_idx, val_idx], 0)
    red_y = np.concatenate([train.Label, test_ds.Label, val.Label], 0)
    feats, cors = dim_reduction_cor(X[red_idx], red_y.astype(np.float32), k=k)
    X = X[:, feats]

    # Torch -------------------------------------------------
    edge_index = torch.from_numpy(edge_index.T)
    edge_index = edge_index.to(torch.long).contiguous()

    X = torch.from_numpy(X).to(torch.float32)
    train_y = torch.tensor(train.Label.astype(int), dtype=torch.float32)
    val_y = torch.tensor(val.Label.astype(int), dtype=torch.float32)
    test_y = torch.tensor(test_ds.Label.astype(int), dtype=torch.float32)

    # ---------------------------------------------------
    print(f'\nNumber of edges in graph: {len(edges)}')
    print(f'Number of features: {X.shape[1]}')
    print(f'Number of nodes in graph: {len(X)}\n')
    print('Using Edge Weights' if edge_weights is not None else 'Not using edge weights')

    if not parse:
        return (edge_index, edge_weights), X, train_ds, test_ds, genes

    return (edge_index, edge_weights), X, (train_idx, train_y), (val_idx, val_y), (test_idx, test_y), genes
