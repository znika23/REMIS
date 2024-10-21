import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

mpl.rcParams['font.family'] = 'Times New Roman'

labels = np.arange(1, 11)
time_reward = np.array([-4650, -3100, -2600, -2200, -1850, -2200, -2100, -2090, -1990, -1800])
time_err = [[268, 152, 281, 47, 106, 203, 235, 103, 153, 84],
            [82, 138, 56, 26, 90, 182, 206, 91, 164, 125]]
for i, err in enumerate(time_err):
    time_err[i] = err / -np.min(time_reward)
time_reward = time_reward / -np.min(time_reward)

acc_reward = np.array([1400, 1350, 850, 1000, 975, 975, 720, 510, 650, 600])
acc_err = [[105, 28, 63, 140, 75, 80, 45, 70, 127, 100],
           [100, 140, 100, 50, 75, 68, 105, 138, 116, 129]]
for i, err in enumerate(acc_err):
    acc_err[i] = err / np.max(acc_reward)
acc_reward = acc_reward / np.max(acc_reward)

# 选择渐变色图
cmap_time = plt.get_cmap('Reds')
cmap_acc = plt.get_cmap('Greens')

norm_time = mcolors.Normalize(vmin=min(np.abs(time_reward)) * 0.2, vmax=max(np.abs(time_reward)) * 2.0)
norm_acc = mcolors.Normalize(vmin=min(np.abs(acc_reward)) * 0.2, vmax=max(np.abs(acc_reward)) * 2.0)
fig, ax = plt.subplots(figsize=(8, 4.5))
width = 0.55


ax.barh(np.arange(len(labels)), time_reward, color=cmap_time(norm_time(np.abs(time_reward))), edgecolor='#C62828', height=width, alpha=0.5, label='Normalized Reward of Time')
ax.errorbar(time_reward, np.arange(len(labels)), xerr=time_err, fmt='none',
            color='#C62828', capsize=5, capthick=2, elinewidth=3, alpha=0.4)
ax.barh(np.arange(len(labels)), acc_reward, color=cmap_acc(norm_acc(np.abs(acc_reward))), edgecolor='#1B5E20', height=width, alpha=0.5, label='Normalized Reward of Accuracy')
ax.errorbar(acc_reward, np.arange(len(labels)), xerr=acc_err, fmt='none',
            color='#1B5E20', capsize=5, capthick=2, elinewidth=3, alpha=0.4)


ax.legend(loc='upper right', frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1.015))

ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-0.5, 9.4)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_xlabel('Normalized Reward in Different MoE Layers of Edge Servers', fontweight='bold', fontsize=12)
ax.set_ylabel('MoE Layers', fontweight='bold', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both', labelsize=10, length=3, direction='in')


plt.tight_layout()
plt.show()
