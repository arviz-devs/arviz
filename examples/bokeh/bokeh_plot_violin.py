"""
Violinplot
==========

_thumb: .2, .8
"""
import arviz as az

data = az.load_arviz_data("non_centered_eight")
ax = az.plot_violin(data, var_names=["mu", "tau"], backend="bokeh")
