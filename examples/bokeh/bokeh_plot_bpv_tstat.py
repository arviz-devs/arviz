"""
Bayesian p-value with median T statistic Posterior plot
=======================================================

_thumb: .6, .5
"""
import arviz as az

data = az.load_arviz_data("regression1d")
ax = az.plot_bpv(data, kind="t_stat", t_stat="0.5", backend="bokeh")
