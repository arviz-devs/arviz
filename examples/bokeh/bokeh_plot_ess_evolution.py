"""
ESS Evolution Plot
==================
"""

import arviz as az

idata = az.load_arviz_data("radon")

ax = az.plot_ess(idata, var_names=["b"], kind="evolution", backend="bokeh")
