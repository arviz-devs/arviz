"""
LOO-PIT Overlay Plot
====================

_thumb: .5, .7
"""
import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("non_centered_eight")

az.plot_loo_pit(idata=idata, y="obs", color="indigo")
