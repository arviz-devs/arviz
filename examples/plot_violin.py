"""
Violinplot
==========

_thumb: .2, .8
"""
import arviz as az
import matplotlib.pyplot as plt

# close all the figures, if open from previous commands
plt.close("all")

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("non_centered_eight")
az.plot_violin(data, var_names=["mu", "tau"])
