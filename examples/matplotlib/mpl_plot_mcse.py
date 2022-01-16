"""
Quantile Monte Carlo Standard Error Plot
========================================

_thumb: .5, .8
_example_title: Quantile Monte Carlo Standard Error Plot
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("centered_eight")
az.plot_mcse(data, var_names=["tau", "mu"], rug=True, extra_methods=True)

plt.show()
