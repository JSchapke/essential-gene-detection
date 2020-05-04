import sys
import argparse
import random
from collections import deque

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics
import sklearn.neural_network
import networkx as nx
sys.path.append('.')
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import tools

class Loss():
    def __init__(self, y):
        self.y = y
        self.pos_mask = y == 1
        self.neg_mask = y == 0

    def __call__(self, out):
        pos_mask = self.pos_mask 
        neg_mask = self.neg_mask 
        loss_p = F.binary_cross_entropy_with_logits(out[pos_mask].squeeze(), self.y[self.pos_mask].cuda())
        loss_n = F.binary_cross_entropy_with_logits(out[neg_mask].squeeze(), self.y[neg_mask].cuda())
        loss = loss_p + loss_n
        return loss

def acc(t1, t2):
    return np.sum(1.0*(t1==t2)) / len(t1)

def main():
    args = tools.get_args()

    scores, roc_aucs = [], []
    for i in range(5):
        seed = i
        print(seed)
        set_seed(seed)
        
        (A, edge_index), X, (train_idx, train_y), (val_idx, val_y), (test_idx, test_y), names = tools.get_data(args, seed=seed)


        if X is None or not X.shape[1]:
            raise ValueError('No features')

        #degrees = A.sum(1).to(torch.float32).reshape((-1, 1))
        #X = torch.cat([X, degrees], dim=1)
        
        in_feats = X.shape[1]
            
        train_x = X[train_idx].cuda()
        val_x = X[val_idx].cuda()
        test_x = X[test_idx].cuda()
        print('train_x', train_x.mean())
        print('test_x', test_x.mean())

        model = nn.Sequential(
                    nn.Linear(in_feats, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1))
        optimizer = torch.optim.Adam(model.parameters())

        lossf = Loss(train_y)
        lossf_val = Loss(val_y)

        model.train()
        model.cuda()
        for i in range(1000):
            out = model(train_x)
            loss = lossf(out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i % 20) == 0:
                model.eval()
                with torch.no_grad():
                    loss_val = lossf_val(model(val_x))
                print('Train loss:', loss.detach().cpu().numpy(), ' |  Val Loss:', loss_val.detach().cpu().numpy())
                model.train()

        model.eval()
        with torch.no_grad():
            out = model(test_x).cpu()
        probs = torch.sigmoid(out).numpy()

        roc_auc = roc_auc_score(test_y, probs)
        roc_aucs.append(roc_auc)

        preds = (probs > 0.5) * 1
        score = acc(preds, test_y)
        scores.append(score)

    print('Acc(all):', scores)
    print('Auc(all):', roc_aucs)
    print('Accuracy:', np.mean(scores))
    print('Auc:', np.mean(roc_aucs))

if __name__ == '__main__':
    main()
