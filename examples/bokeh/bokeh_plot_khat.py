"""
Pareto Shape Plot
=================

_thumb: .7, .5
"""
import arviz as az

idata = az.load_arviz_data("radon")
loo = az.loo(idata, pointwise=True)

ax = az.plot_khat(loo, show_bins=True, backend="bokeh")
