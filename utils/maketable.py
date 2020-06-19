import pandas as pd

df = pd.read_csv('results/human_final.csv', header=None)

methods = ['DC', 'GAT', 'GAT_EXP', 'GAT_EXP_SUB', ['GAT_EXP_ORT', 'GAT_EXP_SUB_ORT']]

#orgs = ['melanogaster', 'yeast', 'coli', 'human']
orgs = ['human' ]
ppis = ['dip', 'biogrid', 'string']

print(methods)
for method in methods:
    for org in orgs:
        for ppi in ppis:
            print('&', end=' ')
            if isinstance(method, list):
                for m in method:
                    rows = df[(df[1] == org) & (df[2] == ppi) & (df[0] == m)].values
                    if len(rows):
                        break
            else:
                rows = df[(df[1] == org) & (df[2] == ppi) & (df[0] == method)].values

            if len(rows):
                print(round(rows[0][-2] * 100, 2), end=' ')
            else:
                print('-', end=' ')


    print('\\\\')
