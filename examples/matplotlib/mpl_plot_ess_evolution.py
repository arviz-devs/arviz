"""
ESS Quantile Plot
=================

_thumb: .2, .8
_example_title: Plot evolution of Effective Sample Size (ESS)
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("radon")

az.plot_ess(idata, var_names=["b"], kind="evolution")

plt.show()
