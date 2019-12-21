"""
Quantile MCSE Errobar Plot
==========================

_thumb: .6, .4
"""
import arviz as az

data = az.load_arviz_data("radon")
ax = az.plot_mcse(data, var_names=["sigma_a"], color="red", errorbar=True, backend="bokeh")
