import torch_geometric as thgeo
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, 
            h_layers=[16, 1],
            in_feats=3,
            dropout=0.2):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        for out_feats in h_layers:
            self.layers.append(GCNConv(in_feats, out_feats))
            in_feats = out_feats

    def forward(self, X, A):
        for layer in self.layers[:-1]:
            X = F.relu(layer(X, A))
            X = F.dropout(X, self.dropout)
        X = self.layers[-1](X, A)
        return X

