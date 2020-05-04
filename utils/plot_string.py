import os
import argparse
import sys; sys.path.append('.')
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from utils import *
import random

parser = argparse.ArgumentParser()
parser.add_argument('--organism', default='yeast')
parser.add_argument('--ppi', default='biogrid')
parser.add_argument('--all_gats', action='store_true')
parser.add_argument('--gat_best', action='store_true')
parser.add_argument('--gat_ppi', action='store_true')
parser.add_argument('--methods', action='store_true')
args = parser.parse_args()

plt.style.use('ggplot')
plt.figure(figsize=(10, 4))

def org_name(s):
    yeast = 'string_0.{}'
    human = 'human_string_0.{}'
    coli = 'coli_string_0.{}'

    if s == yeast:
        return 'Yeast'
    if s == coli:
        return 'Coli'
    if s == human:
        return 'Human'

def main(res):
    yeast = 'string_0.{}'
    human = 'human_string_0.{}'
    coli = 'coli_string_0.{}'

    colors = ['#d62728', '#9467bd', '#1f77b4']

    orgs = [(yeast, [], []), (human, [], []), (coli, [], [])]
    
    for i in range(3, 10):
        for (org, means, stds) in orgs:
            name = org.format(i)
            mask = res['Model Type'] == name
            if np.any(mask):
                mean = res[mask]['Mean'].values[0]
                std = res[mask]['Std Dev'].values[0]
                means.append(mean)
                stds.append(std)
            else:
                means.append(np.NaN)
                stds.append(np.NaN)

    t = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    i = 0
    for (org, means, stds) in orgs:
        means=  np.array(means)
        stds=  np.array(stds)

        name = org_name(org)
        plt.plot(t, means, linewidth=4, color=colors[i])
        plt.fill_between(t, means-stds, means+stds, alpha=0.2, facecolor=colors[i]) #, edgecolor='#CC4F1B', facecolor='#FF9848')
        if name == 'Coli':
            xytext=(10,5)
        else:
            xytext=(10,-2)

        plt.annotate(name, xy=(t[-1],means[-1]), xytext=xytext, textcoords="offset points", size=14, va="center", color=colors[i])
        i += 1

#xycoords = plt.get_yaxis_transform(), #color=line.get_color(),

    plt.title('Performance on String DB')
    plt.xlabel('Threshold')
    plt.ylabel('ROC AUC', labelpad=10)
    
    plt.savefig('plots/string.pdf')

if __name__ == '__main__':

    path = '/home/schapke/projects/research/2020_FEB_JUN/src/results/results.csv'
    res = pd.read_csv(path)
    main(res)
