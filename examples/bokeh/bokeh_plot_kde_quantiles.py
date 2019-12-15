"""
KDE quantiles Bokeh
===================

_thumb: .2, .8
"""
import arviz as az
import numpy as np

dist = np.random.beta(np.random.uniform(0.5, 10), 5, size=1000)
ax = az.plot_kde(dist, quantiles=[0.25, 0.5, 0.75], backend="bokeh")
