"""
LOO-PIT ECDF Plot
=================

_thumb: .5, .7
"""
import arviz as az

idata = az.load_arviz_data("radon")

ax = az.plot_loo_pit(idata, y="y", ecdf=True, color="orange", backend="bokeh")
