import sys; sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

from runners.run_mlp import run
from utils import *
import tools


class Estimator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(X, y):
        print('test')
        x = X[self.idx]
        return self.func(x, y, **self.kwargs)

    def _get_tags(self):
        return {
            'non_deterministic': False, 
            'requires_positive_X': False, 
            'requires_positive_y': False, 
            'X_types': ['2darray'], 
            'poor_score': False, 
            'no_validation': False, 
            'multioutput': False, 
            'allow_nan': False, 
            'stateless': False, 
            'multilabel': False, 
            '_skip_test': False, 
            'multioutput_only': False, 
            'binary_only': False, 
            'requires_fit': True
        }


def cor_selector(X, y,num_feats):
    cors = np.zeros((X.shape[1])) 

    # calculate the correlation with y for each feature
    for i in range(X.shape[1]):
        cor = np.corrcoef(X[:, i], y)[0, 1]
        if not np.isnan(cor):
            cors[i] = cor

    features = np.zeros_like(cors).astype(bool)
    features[np.argsort(-cors)[:num_feats]] = True
    return features, cors


def main(args):
    seed = np.random.randint(0, 1000000)
    _, X, (idx1, y1), (idx2, y2), (idx3, y3), names = tools.get_data(args.__dict__, seed=seed)
    idx = np.concatenate([idx1, idx2, idx3], 0)
    y = np.concatenate([y1, y2, y3], 0)
    X = X[idx]

    feats, cors = cor_selector(X, y, 50)
    print(cors)

#    # Plot number of features VS. cross-validation scores
#    plt.figure()
#    plt.xlabel("Number of features selected")
#    plt.ylabel("Cross validation score (nb of correct classifications)")
#    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#    plt.show()


if __name__ == '__main__':
    args = tools.get_args()
    main(args)
