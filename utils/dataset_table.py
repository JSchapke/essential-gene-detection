import sys; sys.path.append('.')
import numpy as np
from pprint import pprint

from utils import data


if __name__ == '__main__':

    organisms = ["yeast", "coli", "melanogaster", "human"]
    ppis = ["biogrid", "string", "dip"]

    info = {}

    for organism in organisms:
        for ppi in ppis:
            args = dict(organism=organism, ppi=ppi,
                 expression=True, orthologs=True, sublocalizations=True if organism != 'coli' else False,
                 string_thr=500
            )

            # Getting the data ----------------------------------
            (edges, edge_weights), X, train_ds, test_ds, genes = data(**args)
            print('Fetched data', ppi, organism)
            # ---------------------------------------------------

            n_labels = len(train_ds) + len(test_ds)
            n_positives = (test_ds[:, 1] == 1).sum() + (train_ds[:, 1] ==1).sum()
            n_negatives = (test_ds[:, 1] == 0).sum() + (train_ds[:, 1] ==0).sum()
            assert n_labels == n_positives + n_negatives

            key = f'{organism}_{ppi}'
            value = dict(
                    number_of_genes=len(genes),
                    number_of_genes_ppi=len(np.unique(edges)),
                    number_of_edges_ppi=len(edges),
                    number_of_genes_feats=X.shape[0],
                    number_of_labeled_genes=n_labels,
                    number_of_labeled_test_genes=len(test_ds),
                    number_of_positive_labeled_genes=n_positives,
                    number_of_negative_labeled_genes=n_negatives,
            )
            info[key] = value

            print('='*50)
            print(key)
            pprint(value)


    for organism in organisms:
        _ppis = [p.capitalize() for p in ppis]
        print(organism.capitalize(), '&', ' & '.join(_ppis), '\\\\')
        keys = [f'{organism}_{ppi}' for ppi in ppis]

        n_nodes = [info[k]['number_of_genes_ppi'] for k in keys]
        n_nodes = [str(v) for v in n_nodes]
        print('N. nodes &', ' & '.join(n_nodes), '\\\\')

        n_edges = [info[k]['number_of_edges_ppi'] for k in keys]
        n_edges = [str(v) for v in n_edges]
        print('N. edges &', ' & '.join(n_edges), '\\\\')

        n_labels = [info[k]['number_of_labeled_genes'] for k in keys]
        n_labels = [str(v) for v in n_labels]
        print('N. labeled edges &', ' & '.join(n_labels), '\\\\')

        n_positives = [info[k]['number_of_positive_labeled_genes'] for k in keys]
        n_positives = [str(v) for v in n_positives]
        print('N. positive labels &', ' & '.join(n_positives), '\\\\')

        n_test = [info[k]['number_of_labeled_test_genes'] for k in keys]
        n_test = [str(v) for v in n_test]
        print('N. test labels &', ' & '.join(n_test), '\\\\')
