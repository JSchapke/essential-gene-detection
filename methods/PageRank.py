import os
import sys
import numpy as np
import networkx as nx
sys.path.append('.')
from utils import *

class PageRank:
    def __init__(self, G, train):
        self.G = G
        self.back_propagation = 0.3
        self.roots = train[train[:, 1] == 1][:, 0]
        self.n_roots = len(self.roots)

        nodes = self.G.nodes()
        self.iter = 20

    def rank(self):
        probs = {}

        for node in self.G.nodes():
            if node in self.roots:
                probs[node] = 1 / self.n_roots
            else:
                probs[node] = 0

        for i in range(self.iter):
            for node in self.G.nodes():
                if node in self.roots:
                    prior = 1 / self.n_roots
                else:
                    prior = 0

                transition_prob = 0
                for neighbor in list(self.G.neighbors(node)):
                    transition_prob +=  probs[neighbor] / nx.degree(self.G)[neighbor]
                    
                prob = (1 - self.back_propagation) * transition_prob + self.back_propagation * prior
                probs[node] = prob

        return probs


if __name__ == '__main__':
    G, train, test = yeast_data()
    root = '/home/schapke/projects/research/2020_FEB_JUN/src/results/yeast_final'
    path = os.path.join(root, 'PageRank')
    update = True

    if not os.path.isfile(path) or update:
        pg = PageRank(G, train)
        probs = pg.rank().items()
        probs = np.array(list(probs))
        np.save(path, probs)


