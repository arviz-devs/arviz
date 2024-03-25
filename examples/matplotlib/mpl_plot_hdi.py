"""
Plot HDI
========
_gallery_category: Regression or Time Series
"""

import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-doc")

x_data = np.random.normal(0, 1, 100)
y_data = 2 + x_data * 0.5
y_data_rep = np.random.normal(y_data, 0.5, (4, 200, 100))
ax = az.plot_hdi(x_data, y_data_rep, plot_kwargs={"ls": "--"})
ax.plot(x_data, y_data)

plt.show()
