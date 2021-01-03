"""
Dist Plot
=========

_thumb: .2, .8
_example_title: Plot distribution.
"""
import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-darkgrid")

a = np.random.poisson(4, 1000)
b = np.random.normal(0, 1, 1000)

_, ax = plt.subplots(1, 2, figsize=(10, 4))
az.plot_dist(a, color="C1", label="Poisson", ax=ax[0])
az.plot_dist(b, color="C2", label="Gaussian", ax=ax[1])

plt.show()
