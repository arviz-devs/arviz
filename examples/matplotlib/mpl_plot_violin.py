"""
Violin plot
===========

_thumb: .2, .8
_example_title: Violin plot
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("non_centered_eight")
az.plot_violin(data, var_names=["mu", "tau"])

plt.show()
