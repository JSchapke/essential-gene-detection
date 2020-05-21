import os
import argparse
import sys; sys.path.append('.')
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from utils import *
import random

yeast_df = pd.read_csv('results/yeast_final.csv', header=None)
coli_df = pd.read_csv('results/coli_final.csv', header=None)
human_df = pd.read_csv('results/human_final.csv', header=None)
fly_df = pd.read_csv('results/melanogaster_final.csv', header=None)

plt.style.use('ggplot')
plt.figure(figsize=(10, 4))

t = np.arange(0.1, 1, .1)
colors = ['#1f77b4', '#9467bd', '#d62728', '#7f7f7f']

def plot_org(df, entry, name, idx):
    mask = [entry in v for v in df[0].values]
    means = np.array([np.NaN] * 9)
    stds = np.array([np.NaN] * 9)
    for i, row in df[mask].iterrows():
        n, _, _, _, _, _, _, mean, std = row
        n = int(n[-1])  -1
        means[n] = mean
        stds[n] = std

    plt.plot(t, means, linewidth=4, color=colors[idx])
    plt.fill_between(t, means-stds, means+stds, alpha=0.2, facecolor=colors[idx]) #, edgecolor='#CC4F1B', facecolor='#FF9848')
    if name == 'Yeast':
        xytext=(10,-6)
    else:
        xytext=(10,-2)
    plt.annotate(name, xy=(t[-1],means[-1]), xytext=xytext, textcoords="offset points", size=14, va="center", color=colors[idx])

plot_org(yeast_df, 'y', 'Yeast', 0)
plot_org(coli_df, 'c', 'Coli', 1)
plot_org(human_df, 'h', 'Human', 2)
plot_org(fly_df, 'm', 'Fly', 3)


plt.title('Performance on STRING')
plt.xlabel('Filter threshold')
plt.ylabel('ROC AUC', labelpad=10)
plt.savefig('plots/string.pdf')
plt.show()


def main(res):
    colors = ['#d62728', '#9467bd', '#1f77b4', '#7f7f7f']

    for (org, means, stds) in orgs:

        name = org_name(org)
        plt.plot(t, means, linewidth=4, color=colors[i])
        plt.fill_between(t, means-stds, means+stds, alpha=0.2, facecolor=colors[i]) #, edgecolor='#CC4F1B', facecolor='#FF9848')

        plt.annotate(name, xy=(t[-1],means[-1]), xytext=xytext, textcoords="offset points", size=14, va="center", color=colors[i])
        i += 1

#xycoords = plt.get_yaxis_transform(), #color=line.get_color(),

