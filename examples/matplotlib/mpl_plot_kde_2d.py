"""
2d KDE
======

_thumb: .1, .8
"""
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

az.style.use("arviz-darkgrid")

az.plot_kde(np.random.rand(100), np.random.rand(100))

plt.show()
