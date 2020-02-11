"""
2d KDE Bokeh
============

_thumb: .1, .8
"""
import arviz as az
import numpy as np

ax = az.plot_kde(np.random.rand(100), np.random.rand(100), backend="bokeh")
