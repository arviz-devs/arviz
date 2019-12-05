"""
ESS Quantile Plot
=================

_thumb: .2, .8
"""
import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("centered_eight")

az.plot_ess(idata, kind="evolution")
