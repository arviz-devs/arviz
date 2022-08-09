"""
ESS Quantile Plot
=================
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("radon")

az.plot_ess(idata, var_names=["sigma"], kind="quantile", color="C4")

plt.show()
