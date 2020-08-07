"""
Bayesian p-value Posterior plot
===============================

_thumb: .6, .5
"""
import arviz as az

data = az.load_arviz_data("regression1d")
ax = az.plot_bpv(data, backend="bokeh")
