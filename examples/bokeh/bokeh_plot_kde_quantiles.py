"""
KDE quantiles Bokeh
===================
"""

import numpy as np

import arviz as az

dist = np.random.beta(np.random.uniform(0.5, 10), 5, size=1000)
ax = az.plot_kde(dist, quantiles=[0.25, 0.5, 0.75], backend="bokeh")
