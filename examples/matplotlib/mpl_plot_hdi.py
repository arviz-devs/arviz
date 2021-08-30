"""
Plot HDI
========

_thumb: .8, .8
"""
import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-darkgrid")

x_data = np.random.normal(0, 1, 100)
y_data = 2 + x_data * 0.5
y_data_rep = np.random.normal(y_data, 0.5, (4, 200, 100))
az.plot_hdi(x_data, y_data_rep, color="k", plot_kwargs={"ls": "--"})
plt.plot(x_data, y_data, "C6")

plt.show()
