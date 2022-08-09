"""
ESS Quantile Plot
=================
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("radon")

az.plot_ess(idata, var_names=["b"], kind="evolution")

plt.show()
