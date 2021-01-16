"""
Quantile MCSE Errobar Plot
==========================

_thumb: .6, .4
_example_title: Quantile MCSE plot with errobars
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("radon")
az.plot_mcse(data, var_names=["sigma_a"], color="C4", errorbar=True)

plt.show()
