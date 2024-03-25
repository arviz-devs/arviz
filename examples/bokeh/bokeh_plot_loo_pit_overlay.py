"""
LOO-PIT Overlay Plot
====================
"""

import arviz as az

idata = az.load_arviz_data("non_centered_eight")

ax = az.plot_loo_pit(idata=idata, y="obs", color="green", backend="bokeh")
