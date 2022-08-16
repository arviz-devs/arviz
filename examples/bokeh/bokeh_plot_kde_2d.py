"""
2d KDE (default style)
======================
"""
import numpy as np

import arviz as az

ax = az.plot_kde(np.random.rand(100), np.random.rand(100), backend="bokeh")
