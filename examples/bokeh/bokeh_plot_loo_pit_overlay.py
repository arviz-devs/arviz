"""
LOO-PIT Overlay Plot
====================

_thumb: .5, .7
"""
import arviz as az

idata = az.load_arviz_data("non_centered_eight")

ax = az.plot_loo_pit(idata=idata, y="obs", color="green", backend="bokeh")
