import sys; sys.path.append('.')

from sklearn.metrics import roc_auc_score
from torch_geometric.nn.models import Node2Vec
from sklearn.svm import SVC

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from runners.run_mlp import mlp_fit_predict
from utils.utils import *
from runners import tools

PARAMS = { 
        'embedding_dim': 64,
        'walk_length': 8,
        'context_size': 8,
        'walks_per_node': 2,
        'num_negative_samples': 1,
} 
LR = 1e-2
WEIGHT_DECAY = 5e-4
EPOCHS=100
DEV = torch.device('cuda')

def run():

        print('Training Node2Vec')

        num_nodes = len(np.unique(edge_index.reshape((-1))))
        train_nodes = torch.arange(0, int(num_nodes * 0.9)).long()
        val_nodes = torch.arange(int(num_nodes * 0.9), num_nodes).long()
        idx = np.arange(train_nodes.shape[0])

        model = Node2Vec(edge_index, **PARAMS).to(DEV)
        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        patience, best_loss = 10, np.Inf
        steps = 0

        for i in range(EPOCHS):
            train_loss = 0
            model.train()
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(DEV), neg_rw.to(DEV))
                loss.backward()
                optimizer.step()
                train_loss += loss.data.item()

            model.eval()
            with torch.no_grad():
                z = model()
                acc = model.test(z[train_mask], train_y, z[test_mask], test_y, max_iter=150)
            #print(f'{i}. Train loss:', train_loss, ' |  Val Loss:', val_loss, end='\r')
            print(f'Epoch {i}. Acc: {acc}', end='\r')

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
        print('Training SVM head.')
        svm = SVC(class_weight='balanced', random_state=0)
        svm.fit(X, y)
        return svm.predict(test_x)


    def fit_predict(self, edge_index, X, y, idx, test_x, test_y, test_idx):
        self.train_n2v(edge_index, y, idx, test_y, test_idx)

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
    probs = model.fit_predict(edge_index, x, y, idx, test_x, test_y, test_idx)
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
    print(mean, std)

    name = get_name(args)


    #df_path = 'results/results.csv'
    #df = pd.read_csv(df_path)

    #df.loc[len(df)] = [name, args.organism, args.ppi, args.expression, args.orthologs, args.sublocs, args.n_runs, mean, std]
    #df.to_csv(df_path, index=False)
    #print(df.tail())

