"""
Pareto Shape Plot
=================
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("radon")
loo = az.loo(idata, pointwise=True)

az.plot_khat(loo, show_bins=True)

plt.show()
