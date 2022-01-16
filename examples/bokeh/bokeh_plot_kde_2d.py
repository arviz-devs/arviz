"""
2d KDE (default style)
======================

_thumb: .1, .8
"""
import numpy as np

import arviz as az

ax = az.plot_kde(np.random.rand(100), np.random.rand(100), backend="bokeh")
