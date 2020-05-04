import random
from collections import deque

import networkx as nx
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import optuna
from sklearn.model_selection import train_test_split

from models.gcn.gcn_pytorch import GCN
from utils import biomarker_dataset, set_seed, evaluate

LR = 1e-3
WEIGHT_DECAY = 1e-4

def train(X, A, train_y, train_mask, val_y, val_mask):
    model = GCN(in_feats=X.shape[1])
    model.cuda()
    X, A = X.cuda(), A.cuda()

    train_y = torch.tensor(train_y, dtype=torch.float32)
    val_y = torch.tensor(val_y, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fnc = nn.BCEWithLogitsLoss()

    val_acc, val_steps = 0, 250
    train_acc = 0
    iterable = tqdm(range(5000)) 
    for i in iterable:
        model.train()
        logits = model(X, A)

        idxs = torch.tensor(np.where(train_mask)[0])
        positive_idxs = idxs[train_y == 1]
        negative_idxs = idxs[train_y == 0][:len(positive_idxs)]

        positives = train_y[train_y == 1]
        negatives = train_y[train_y == 0][:len(positives)]
        
#        idxs = range(len(negative_idxs))
#        sample_idx = random.sample(idxs, len(positive_idxs))
#        negative_idxs = negative_idxs[sample_idx]
#        negatives = negatives[sample_idx]

        loss_pos = loss_fnc(logits[positive_idxs].squeeze(), positives.cuda())
        loss_neg = loss_fnc(logits[negative_idxs].squeeze(), negatives.cuda())
        loss = loss_pos + loss_neg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % val_steps == 0:
            train_acc = evaluate(model, X, A, train_y.cuda(), train_mask)
            val_acc = evaluate(model, X, A, val_y.cuda(), val_mask)
        tqdm.set_description(iterable, desc='Loss: %.4f. Train Accuracy %.4f. Validation Accuracy: %.4f' % (loss, train_acc, val_acc), refresh=True)
    return model

def main():
    set_seed(1)

    G, train_names, test_names, train_y, test_y = biomarker_dataset()

    adj = nx.to_numpy_array(G)
    A = np.where(adj == 1)
    A = torch.tensor(A, dtype=torch.long).contiguous()

    nodes = np.array(list(G.nodes))

    train_names, val_names, train_y, val_y = \
            train_test_split(train_names, train_y, test_size=0.1, stratify=train_y, random_state=0)

    train_mask = np.isin(nodes, train_names)
    val_mask   = np.isin(nodes, val_names)
    test_mask  = np.isin(nodes, test_names)

    X = pd.read_csv('X.csv').values
    X = (X - X.mean(0)) / X.std(0)
    X = torch.tensor(X, dtype=torch.float32)

    model = train(X, A, train_y, train_mask, val_y, val_mask)

    model.eval()
    with torch.no_grad():
        logits = model(X.cuda(), A.cuda())
    probs = torch.sigmoid(logits)
    probs = probs.cpu().numpy()
    probs = probs[test_mask]

    for i, prob in enumerate(probs):
        print(prob)
        if i == 10:
            break
    print(probs.min(), probs.mean(), probs.max())

    path = f'models/methods/results/GCN'
    np.save(path, probs)
    print(f'Saved results to: {path}')

if __name__ == '__main__':
    main()
