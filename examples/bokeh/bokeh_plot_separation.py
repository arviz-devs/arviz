"""
Separation Plot
===============

_thumb: .2, .8
"""
import arviz as az

idata = az.load_arviz_data("classification10d")

ax = az.plot_separation(idata=idata, y="outcome", y_hat="outcome", figsize=(8, 1), backend="bokeh")
