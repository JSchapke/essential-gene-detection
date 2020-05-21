import sys; sys.path.append('.')

from sklearn.metrics import roc_auc_score
from torch_geometric.nn.models import Node2Vec
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from runners.run_mlp import mlp_fit_predict
from utils import *
import tools

#PARAMS = { 
#        'embedding_dim': 32,
#        'walk_length': 64,
#        'context_size': 64,
#        'walks_per_node': 2 } 
PARAMS = { 
        'embedding_dim': 64,
        'walk_length': 8,
        'context_size': 8,
        'walks_per_node': 2 } 
LR = 1e-2
WEIGHT_DECAY = 5e-4
EPOCHS=100


class Model:
    def __init__(self, head_type='svm'):
        self.head_type = head_type

    def train_n2v(self, edge_index):
        num_nodes = len(np.unique(edge_index.reshape((-1))))
        train_nodes = torch.arange(0, int(num_nodes * 0.8)).to(torch.long)
        train_nodes = torch.arange(0, num_nodes).to(torch.long)
        val_nodes = torch.arange(int(num_nodes * 0.8), num_nodes).to(torch.long)

        model = Node2Vec(num_nodes, **PARAMS)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        patience, best_loss = 10, np.Inf
        steps = 0

        for i in range(EPOCHS):
            train_loss = model.loss(edge_index, train_nodes)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                val_loss = model.loss(edge_index, val_nodes)
            print(f'{i}. Train loss:', train_loss.detach().cpu().numpy(),
                   ' |  Val Loss:', val_loss.detach().cpu().numpy(), end='\r')

            if val_loss < best_loss:
                best_loss = val_loss
                steps = 0
            else:
                steps += 1
                if steps == patience:
                    break
        print('')

        self.embedding = model.embedding.weight.detach().cuda()
        print('self.embedding.shape', self.embedding.shape)


    def svm_fit_predict(self, X, y, test_x):
        svm = SVC(class_weight='balanced', random_state=0)
        svm.fit(X, y)
        return svm.predict(test_x)


    def fit_predict(self, edge_index, X, y, idx, test_x, test_idx):
        print(self.head_type)
        self.train_n2v(edge_index)

        embedding = self.embedding
        print('Embedding.shape:', embedding.shape)

        train_embedding = embedding[idx]
        test_embedding = embedding[test_idx]

        #X = torch.cat([train_embedding, X], dim=1)
        #test_x = torch.cat([test_embedding, test_x], dim=1)
        X = train_embedding
        test_x = test_embedding
        print('X.shape:', X.shape)
        print('test_x.shape:', test_x.shape)

        if self.head_type == 'svm':
            X = X.cpu().numpy()
            y = y.cpu().numpy()
            test_x = test_x.cpu().numpy()
            return self.svm_fit_predict(X, y, test_x)
        elif self.head_type == 'mlp':
            return mlp_fit_predict(X, y, test_x)




def run(head_type, x, y, idx, test_x, test_y, test_idx, edge_index):
    model = Model(head_type)
    probs = model.fit_predict(edge_index, x, y, idx, test_x, test_idx)
    print(probs)
    roc_auc = roc_auc_score(test_y, probs)
    print('Roc auc:', roc_auc)
    return roc_auc


def main(args):

    roc_aucs = []
    for i in range(args.n_runs):
        seed = i
        set_seed(seed)
        
        (edge_index, _), X, (train_idx, train_y), (val_idx, val_y), (test_idx, test_y), names = tools.get_data(args.__dict__, seed=seed)

        if X is None or not X.shape[1]:
            raise ValueError('No features')
            
        train_idx = train_idx + val_idx
        train_x = X[train_idx].cuda()
        train_y = train_y.tolist() + val_y.tolist()
        train_y = torch.tensor(train_y).cuda()
        test_x = X[test_idx].cuda()

        print(len(np.unique(edge_index.reshape((-1)))))
        print(len(train_idx), len(test_x))

        roc_auc = run(args.head_type, train_x, train_y, train_idx, test_x, test_y, test_idx, edge_index)

        roc_aucs.append(roc_auc)


    print('Auc(all):', roc_aucs)
    print('Auc:', np.mean(roc_aucs))

    return np.mean(roc_aucs), np.std(roc_aucs)

def get_name(args):
    if args.name:
        return args.name

    name = 'N2V_' + args.head_type.upper()
    if args.no_ppi:
        name += '_NO-PPI'
    if args.expression:
        name += '_EXP'
    if args.sublocs:
        name += '_SUB'
    if args.orthologs:
        name += '_ORT'

    return name


if __name__ == '__main__':
    parser = tools.get_args(parse=False)
    parser.add_argument('--head_type', default='svm', help='Head for the two step model ["svm", "mlp"]')
    args = parser.parse_args()

    print('Head:', args.head_type)

    mean, std = main(args)

    name = get_name(args)


    df_path = 'results/results.csv'
    df = pd.read_csv(df_path)

    df.loc[len(df)] = [name, args.organism, args.ppi, args.expression, args.orthologs, args.sublocs, args.n_runs, mean, std]
    df.to_csv(df_path, index=False)
    print(df.tail())

