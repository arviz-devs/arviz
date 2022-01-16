"""
Rank Parallel Plot
==================

_thumb: .2, .5
"""
import arviz as az

data = az.load_arviz_data("centered_eight")
ax = az.plot_parallel(data, var_names=["theta", "tau", "mu"], norm_method="rank", backend="bokeh")
