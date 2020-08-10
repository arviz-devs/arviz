"""
Traceplot Bokeh
===============

_thumb: .1, .8
"""
import arviz as az

data = az.load_arviz_data("non_centered_eight")
ax = az.plot_trace(data, var_names=("tau", "mu"), backend="bokeh")
