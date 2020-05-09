import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('results/results.csv')


organisms = ['yeast', 'coli', 'human']
ppi = ['biogrid', 'string', 'dip']
_methods = [['DC', 'LAC', 'NC', 'GAT_EXP_SUB_ORT'], ['DC', 'LAC', 'NC', 'GAT_EXP_ORT'], ['DC', 'LAC', 'NC', 'GAT_EXP_SUB_ORT']]


plt.style.use('ggplot')
fig = plt.figure(figsize=(10, 8))
#fig.suptitle('AUC ROC Scores', fontsize=15)


# Main Plot
ax = fig.add_subplot(111)    # The big subplot
ax.set_facecolor('white')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

# Set common labels
ax.set_xlabel('Networks')
ax.set_ylabel('ROC AUC', labelpad=15)

for i, organism in enumerate(organisms):
    ax = fig.add_subplot(2,2, i+1)
    ax.set_title(organism.capitalize(), fontsize=13)

    methods = _methods[i]
    for method in methods:
        rows = df[(df['Organism'] == organism) & df['PPI'].isin(ppi) & (df['Model Type'] == method)]
        rows = rows.sort_values('PPI')
        print(rows, method, organism, ppi)
        rows = rows.iloc[[2,0,1]]
        x_labels = rows['PPI'].values
        x_labels = [x.capitalize() for x in x_labels]

        y_values = rows['Mean'].values
        std = rows['Std Dev'].values

        if 'GAT' in method:
            ax.plot(x_labels, y_values, label='GAT', linewidth=4, color='red')
            ax.fill_between(x_labels, y_values-std, y_values+std, color='red', alpha=0.2) #, edgecolor='#CC4F1B', facecolor='#FF9848')
        else:
            ax.plot(x_labels, y_values, label=method, linewidth=4)
            ax.fill_between(x_labels, y_values-std, y_values+std, alpha=0.2) #, edgecolor='#CC4F1B', facecolor='#FF9848')

    ax.legend(loc='best')

fig.subplots_adjust(top=0.90, bottom=0.12, left=0.07, right=1, hspace=0.3)
plt.suptitle('AUC ROC Scores')
plt.savefig(f'plots/aucs.pdf')
#plt.show()
