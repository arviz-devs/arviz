"""
ESS Quantile Plot
=================

_thumb: .4, .5
"""
import matplotlib.pyplot as plt
import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("radon")

az.plot_ess(idata, var_names=["sigma_y"], kind="quantile", color="C4")

plt.show()
