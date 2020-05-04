
# (coli, string)
gat_huge = {
    'lr': 0.005,
    'weight_decay': 3e-4,
    'h_feats': [128, 1], 
    'heads': [8, 1],  
    'dropout': 0.4,
    'negative_slope': 0.2 }

# (coli, biogrid) 
gat_large = {
    'lr': 0.002,
    'weight_decay': 3e-4,
    'h_feats': [64, 1], 
    'heads': [8, 1],  
    'dropout': 0.4,
    'negative_slope': 0.2 }

# coli / yeast
gat_medium = {
    'lr': 0.005,
    'weight_decay': 1e-4,
    'h_feats': [16, 1], 
    'heads': [8, 1],  
    'dropout': 0.1,
    'negative_slope': 0.2 }

# (coli, dip) (yeast, dip) (yeast, biogrid) (yeast, string)
# humans / yeast
gat_small = {
    'lr': 0.005,
    'weight_decay': 5e-4,
    'h_feats': [8, 1], 
    'heads': [8, 1],  
    'dropout': 0.4,
    'negative_slope': 0.2 }


gat_simpler = {
    'lr': 0.01,
    'weight_decay': 5e-4,
    'h_feats': [16, 1], 
    'heads': [1, 1],  
    'dropout': 0.6,
    'negative_slope': 0.2 }

