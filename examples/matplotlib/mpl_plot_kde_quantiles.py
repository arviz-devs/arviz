"""
KDE quantiles
=============

_thumb: .2, .8
_example_title: Plot KDE of quantiles
"""
import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-darkgrid")

dist = np.random.beta(np.random.uniform(0.5, 10), 5, size=1000)
az.plot_kde(dist, quantiles=[0.25, 0.5, 0.75])

plt.show()
