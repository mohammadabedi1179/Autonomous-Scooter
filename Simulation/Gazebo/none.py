import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects

plt.rcParams.update({
    'figure.facecolor': '#FEFEFE',
    'axes.facecolor': '#F9F9F9',
    'axes.edgecolor': '#D0D0D0',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.color': '#E8E8E8',
    'grid.linewidth': 0.8,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})



# Example plot
x = np.linspace(0, 10, 1000)
y1 = np.sin(x) * np.exp(-x/10)
y2 = np.cos(x) * np.exp(-x/10)

# Soft pastel colors
soft_colors = ['#8ECAE6', '#FFB4A2', '#B5E48C', '#E9C46A', '#D4A5A5']

fig, ax = plt.subplots(figsize=(10, 6))
for i, (y, label) in enumerate([(y1, 'A'), (y2, 'B')]):
    ax.plot(x, y, color=soft_colors[i], linewidth=2.5, 
            label=f'Signal {label}', alpha=0.85)
    ax.fill_between(x, y, alpha=0.15, color=soft_colors[i])

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Soft Pastel Theme')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

