"""
2d KDE
======

_thumb: .1, .8
"""
import arviz as az
import numpy as np

az.style.use('arviz-darkgrid')

az.plot_kde(np.random.rand(100), np.random.rand(100))
