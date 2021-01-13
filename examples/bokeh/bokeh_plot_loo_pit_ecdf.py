"""
LOO-PIT ECDF Plot
=================

_thumb: .5, .7
_example_title: Plot LOO predictive ECDF compared to ECDF of uniform distribution to assess predictive calibration.
"""
import arviz as az

idata = az.load_arviz_data("radon")

ax = az.plot_loo_pit(idata, y="y", ecdf=True, color="orange", backend="bokeh")
