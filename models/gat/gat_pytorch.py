import torch.nn.functional as F
import torch.nn as nn
import torch

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))

        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)


    def forward(self, x, edge_index, size=None, edge_weights=None, return_alpha=False):
        self.return_alpha = return_alpha

        if size is None and torch.is_tensor(x):
            edge_index, edge_weights = remove_self_loops(edge_index, edge_weights)
            edge_index, edge_weights = add_self_loops(edge_index, edge_weight=edge_weights, 
                                                            num_nodes=x.size(self.node_dim))

            self.edge_weights = edge_weights

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        if self.return_alpha:
            return self.propagate(edge_index, size=size, x=x), self.alpha, edge_index

        return self.propagate(edge_index, size=size, x=x)


    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        if self.return_alpha:
            self.alpha = alpha #.detach().cpu().numpy()

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if self.edge_weights is not None:
            edge_weights = self.edge_weights.reshape((-1, 1, 1))
            return x_j * alpha.view(-1, self.heads, 1) * edge_weights

        return x_j * alpha.view(-1, self.heads, 1)


    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GAT(nn.Module):
    def __init__(self, in_feats=1, 
            h_feats=[8, 8, 1], 
            heads=[8, 8, 4],  
            dropout=0.6,
            negative_slope=0.2,
            **kwargs):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for i, h_feat in enumerate(h_feats):
            last = i + 1 == len(h_feats)
            self.layers.append(GATConv(in_feats, h_feat, 
                                            heads=heads[i], 
                                            dropout=dropout,
                                            concat=False if last else True))
            in_feats = h_feat * heads[i]

    def forward(self, X, A, edge_weights=None, return_alphas=False):
        alphas = []
        for layer in self.layers[:-1]:
            if return_alphas:
                X, alpha, _ = layer(X, A, edge_weights=edge_weights, return_alpha=True)
                alphas.append(alpha)
            else:
                X = layer(X, A, edge_weights=edge_weights)
            X = F.relu(X)
            X = F.dropout(X, self.dropout)

        if return_alphas:
            X, alpha, edge_index = self.layers[-1](X, A, edge_weights=edge_weights, return_alpha=True)
            alphas.append(alpha)
            return X, alphas, edge_index
        
        X = self.layers[-1](X, A, edge_weights=edge_weights)
        return X



