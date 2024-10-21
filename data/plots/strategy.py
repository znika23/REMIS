import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

mpl.rcParams['font.family'] = 'Times New Roman'
# task_acc_rate = [[0.02, 0.04, 0.04, 0.16, 0.26, 0.48, 0.66, 0.74, 0.88, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                  [0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.08, 0.14, 0.24, 0.32, 0.46, 0.68, 0.84, 0.90, 0.96, 0.98, 1],
#                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.38, 0.76, 0.84, 0.94, 1],
#                  [0, 0.04, 0.08, 0.28, 0.44, 0.62, 0.84, 0.92, 0.96, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

x = np.arange(0.05, 1.05, 0.05)
data1 = [0.02, 0.02, 0.04, 0.1, 0.1, 0.2, 0.18, 0.08, 0.14, 0.08, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0]
data2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.08, 0.06, 0.1, 0.08, 0.14, 0.22, 0.16, 0.06, 0.06, 0.02, 0.02]
data3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.18, 0.38, 0.08, 0.1, 0.06]
data4 = [0, 0.04, 0.04, 0.2, 0.16, 0.18, 0.22, 0.08, 0.04, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
task_time = np.array([1.95, 1.68, 2.05, 1.43])
task_time = task_time / np.max(task_time)
labels = ['General Top-2 Activation', 'MoE', 'Inference Accuracy Greedy', 'Inference Time Greedy']

fig, axs = plt.subplots(4, 1, figsize=(8, 7))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
edge_colors = ['#1565C0', '#FF6F20', '#1B5E20', '#C62828']

for i, data in enumerate([data1, data2, data3, data4]):
    axs[i].bar(x, data, color=colors[i], edgecolor=edge_colors[i], linewidth=2.0, width=0.04, alpha=0.7)
    axs_twin = axs[i].twinx()
    axs_twin.tick_params(axis='both', which='both', labelsize=12, length=3, direction='in')
    axs_twin.set_ylim(0.6, 1.1)
    axs_twin.set_ylabel('Average Task Time', fontweight='bold', fontsize=10)
    axs_twin.axhline(y=task_time[i], color=colors[i], linewidth=3, alpha=0.4, linestyle='-.')

    axs[i].set_title(labels[i], fontsize=12)
    axs[i].set_ylabel('Probability Distribution', fontweight='bold', fontsize=10)
    axs[i].tick_params(axis='both', which='both', labelsize=12, length=3, direction='in')
    axs[i].xaxis.set_minor_locator(MultipleLocator(0.05))
    axs[i].grid(linestyle='--', alpha=0.5, which='both')

axs[3].set_xlabel('Task Accuracy at MNR = 0.25 and MAP = 0.2', fontweight='bold', fontsize=15)

# 调整布局
plt.tight_layout()
plt.show()