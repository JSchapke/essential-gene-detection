import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

yeast_df = pd.read_csv('outputs/results/yeast_final.csv', header=None)
coli_df = pd.read_csv('outputs/results/coli_final.csv', header=None)
human_df = pd.read_csv('outputs/results/human_final.csv', header=None)
fly_df = pd.read_csv('outputs/results/melanogaster_final.csv', header=None)
print(fly_df.head())

ppi = ['biogrid', 'string', 'dip']
methods = [['DC', 'LAC', 'NC', 'GAT_EXP_SUB_ORT'], ['DC', 'LAC', 'NC', 'GAT_EXP_ORT'], [
    'DC', 'LAC', 'NC', 'GAT_EXP_SUB_ORT'], ['DC', 'LAC', 'NC', 'GAT_EXP_SUB_ORT']]


plt.style.use('ggplot')
fig = plt.figure(figsize=(10, 8))

# Main Plot
ax = fig.add_subplot(111)    # The big subplot
ax.set_facecolor('white')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False,
               bottom=False, left=False, right=False)

colors = ['#E24A33', '#348ABD', '#FBC15E', '#8EBA42', '#FFB5B8', '#988ED5', ]

def plot(ax, df, methods):
    i = 0
    for method in methods:
        rows = df[df[0] == method]
        rows = rows.sort_values(2)
        rows = rows.iloc[[2, 0, 1]]

        x_labels = rows[2].values
        x_labels = [x.capitalize() for x in x_labels]

        y_values = rows[7].values.astype(float)
        std = rows[8].values.astype(float)

        if 'GAT' in method:
            ax.plot(x_labels, y_values, label='GAT', linewidth=6, color='red')
            # , edgecolor='#CC4F1B', facecolor='#FF9848')
            ax.fill_between(x_labels, y_values-std, y_values +
                            std, color='red', alpha=0.2)
        else:
            ax.plot(x_labels, y_values, label=method, linewidth=6, color=colors[i])
            # , edgecolor='#CC4F1B', facecolor='#FF9848')
            print(y_values, std)
            ax.fill_between(x_labels, y_values-std, y_values+std, alpha=0.2, color=colors[i])
            i += 1

        ax.legend(loc='best')
        ax.set_ylim(top=1, bottom=0.4)


# Set common labels
ax.set_ylabel('ROC AUC', labelpad=15)

ax = fig.add_subplot(221)
ax.set_title('Yeast', fontsize=13)
plot(ax, yeast_df, methods[0])

ax = fig.add_subplot(222)
ax.set_title('Coli', fontsize=13)
plot(ax, coli_df, methods[1])

ax = fig.add_subplot(223)
ax.set_title('Human', fontsize=13)
plot(ax, human_df, methods[2])

ax = fig.add_subplot(224)
ax.set_title('Fly', fontsize=13)
plot(ax, fly_df, methods[3])


fig.subplots_adjust(top=0.90, bottom=0.12, left=0.07, right=1, hspace=0.3)
plt.suptitle('Benchmark on network based methods', fontsize=15)
plt.savefig(f'outputs/plots/aucs2.pdf')
plt.show()
