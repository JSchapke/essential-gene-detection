import sys; sys.path.append('.')
from os.path import join
import torch
import numpy as np
from gat_pytorch import GAT

from utils import *


def main(savepath, predspath, organism, ppi):
    # Data ----------------------------------------
    if organism == 'yeast':
        G, X, train_ds, test_ds = yeast_data(ppi=ppi, expression=True)
    elif organism == 'coli':
        G, X, train_ds, test_ds = coli_data(ppi=ppi, expression=True)
    elif organism == 'human':
        G, X, train_ds, test_ds = human_data(ppi=ppi, expression=True)

    names = np.array(list(G.nodes()), dtype=str)
    ran = range(len(names))
    mapping = dict(zip(names, ran))
    X = (X - X.mean()) / X.std() 
    X = torch.tensor(X, dtype=torch.float32)

    A = np.array(nx.to_numpy_matrix(G))
    A = torch.tensor(A, dtype=torch.float32)
    A = A + np.eye(A.shape[0])
    edge_index = torch.tensor(np.where(A == 1))
    edge_index = edge_index.to(torch.long).contiguous()
    # ---------------------------------------------

    snapshot = torch.load(savepath, map_location='cpu')
    model = GAT(in_feats=X.shape[1], **snapshot['model_params'])
    model.load_state_dict(snapshot['model_state_dict'])

    pred = np.load(predspath)

    train_names, train_y = train_ds[:, 0].tolist(), train_ds[:, 1]
    test_names, test_y = test_ds[:, 0].tolist(), test_ds[:, 1]
    pred_names, pred_y = pred[:, 0], pred[:, 1].astype(np.float32)

    # Node -----------------------------------
    i = -45 # 13 neigs
    max_idx = pred_y.argsort()[i]
    max_name = test_names[max_idx]
    print('Chosen Idx / Prediction Prob / GroundTruth:', max_idx, pred_y[max_idx], test_y[max_idx])
    # ----------------------------------------

    neigs = list(G.neighbors(max_name))
    n = len(neigs)
    no = len(G.nodes)

    model.eval().cpu()
    with torch.no_grad():
        out, alphas, edge_index = model(X, edge_index, return_alphas=True)
    
    I, J = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    layers = len(alphas)
    heads = alphas[0].shape[1]

    
    attention = np.zeros((layers, heads, no, no), dtype=np.float32)
    falpha = np.zeros((no, no))
    for i, alpha in enumerate(alphas):
        print(alpha.shape)
        attention[i, 0, I, J] = alpha[:, 1]
        falpha[I, J] = alpha[:, 0]

    dist = attention[-1, 0, :, mapping[max_name]]
    print('dist.sum():', dist.sum(), '(should be == 1).')

    f = open(f'results/attention_preds_{max_idx}.txt', 'w')
    f.write(f'Name\tInTrain\tClass\tImportance\tPred\n')
    for neig in np.union1d(neigs, [max_name]):
        intrain = False
        id = mapping[neig]
        if neig in test_names:
            cls = test_y[test_names.index(neig)]
        elif neig in train_names:
            intrain = True
            cls = train_y[train_names.index(neig)]
        else:
            cls = -1
        pred = pred_y[max_idx] if neig == max_name else 0
        f.write(f'{neig}\t{intrain}\t{cls}\t{dist[id]}\t{pred}\n')
    f.close()



if __name__ == '__main__':

    organism = 'yeast'
    ppi = 'biogrid'

    ROOT = '/home/schapke/projects/research/2020_FEB_JUN/src'
    weightsdir = join(ROOT, 'models/gat/weights') 
    snapshot_name = f'{organism}_{ppi}_expression'
    weightspath = join(weightsdir, snapshot_name)

    predspath = f'results/{organism}/{ppi}/gat/GAT_EXP.npy'
    predspath = join(ROOT, predspath)

    main(weightspath, predspath, organism, ppi)
