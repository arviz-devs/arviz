"""
ESS Quantile Plot
=================

_thumb: .4, .5
"""
import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("centered_eight")

az.plot_ess(idata, kind="quantile", color="C4")
