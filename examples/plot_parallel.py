"""
Parallel Plot
=============

_thumb: .2, .5
"""
import arviz as az
import matplotlib.pyplot as plt

# close all the figures, if open from previous commands
plt.close("all")

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("centered_eight")
ax = az.plot_parallel(data, var_names=["theta", "tau", "mu"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
