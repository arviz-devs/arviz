"""
Bayesian u-value Plot
=====================
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("regression1d")
az.plot_bpv(data)

plt.show()
