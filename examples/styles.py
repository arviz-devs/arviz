"""
Styles
======

_thumb: .8, .8
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

x = np.linspace(0, 1, 100)
dist = stats.beta(2, 5).pdf(x)

style_list = ['default',
              ['default', 'arviz-colors'],
              'arviz-darkgrid',
              'arviz-whitegrid',
              'arviz-white']

fig = plt.figure(figsize=(12, 12))
for idx, style in enumerate(style_list):
    with az.style.context(style):
        ax = fig.add_subplot(3,2, idx+1, label=idx)
        for i in range(10):
            ax.plot(x, dist - i, f'C{i}', label=f'C{i}')
        ax.set_title(style)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)', rotation=0, labelpad=15)
        ax.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
