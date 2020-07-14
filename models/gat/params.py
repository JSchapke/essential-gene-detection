# gat_yeast = {
#    'lr': 0.005,
#    'weight_decay': 2e-4,
#    'h_feats': [12, 1],
#    'heads': [8, 1],
#    'dropout': 0.3,
#    'negative_slope': 0.2 }


# gat_fly = {
#    'lr': 0.005,
#    'weight_decay': 5e-4,
#    'h_feats': [16, 1],
#    'heads': [1, 1],
#    'dropout': 0.6,
#    'negative_slope': 0.2}

#gat_0 = {
#    'lr': 0.005,
#    'weight_decay': 5e-4,
#    'h_feats': [8, 1],
#    'heads': [8, 1],
#    'dropout': 0.4,
#    'negative_slope': 0.2}

gat_human = {
    'lr': 0.005,
    'weight_decay': 5e-4,
    'h_feats': [16, 1],
    'heads': [4, 1],
    'dropout': 0.4,
    'negative_slope': 0.2,
    'linear_layer': 32
}

gat_yeast = {
    'dropout': 0.20831261071814422, 'h_feats': [16, 1], 'heads': [8, 4], 'lr': 0.000681327728645882, 'weight_decay': 8.013847980434763e-05
}


gat_coli = {'h_feats': [8, 1], 'heads': [4, 4], 'lr': 1.6220675358900915e-05,
            'weight_decay': 1.6474296279432942e-05, 'dropout': 0.21505090213922445}

gat_fly = {'dropout': 0.26196403544471325, 'h_feats': [8, 32, 8, 1], 'heads': [2, 1, 4, 4],
           'lr': 0.0027159751390655232, 'weight_decay': 1.518599263089526e-06}

gat_human = {'h_feats': [32, 1], 'heads': [4, 2], 'lr': 0.0008036075040623076, 'weight_decay': 0.0001625495450304567, 'dropout': 0.3584591410721659 }
