"""
Autocorrelation Plot
====================

_thumb: .8, .8
_example_title: Autocorrelation plot
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("centered_eight")
az.plot_autocorr(data, var_names=("tau", "mu"))

plt.show()
