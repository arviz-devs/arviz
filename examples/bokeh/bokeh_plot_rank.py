"""
Rank plot
=========

_thumb: .1, .8
"""
import arviz as az

data = az.load_arviz_data("centered_eight")
ax = az.plot_rank(data, var_names=("tau", "mu"), backend="bokeh")
