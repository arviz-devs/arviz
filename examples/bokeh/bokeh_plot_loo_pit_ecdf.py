"""
LOO-PIT ECDF Plot
=================
"""

import arviz as az

idata = az.load_arviz_data("radon")

ax = az.plot_loo_pit(idata, y="y", ecdf=True, color="orange", backend="bokeh")
