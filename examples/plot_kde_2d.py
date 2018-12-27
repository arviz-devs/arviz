"""
2d KDE
======

_thumb: .1, .8
"""
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# close all the figures, if open from previous commands
plt.close("all")

az.style.use("arviz-darkgrid")

az.plot_kde(np.random.rand(100), np.random.rand(100))
