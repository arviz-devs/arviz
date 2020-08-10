"""
ESS Quantile Plot
=================

_thumb: .4, .5
"""
import arviz as az

idata = az.load_arviz_data("radon")

ax = az.plot_ess(idata, var_names=["sigma"], kind="quantile", color="red", backend="bokeh")
