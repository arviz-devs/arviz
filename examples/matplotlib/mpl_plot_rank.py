"""
Rank plot
=========

_thumb: .1, .8
_example_title: Rank plot
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("centered_eight")
az.plot_rank(data, var_names=("tau", "mu"))

plt.show()
