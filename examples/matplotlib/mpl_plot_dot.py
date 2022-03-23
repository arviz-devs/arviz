"""
Dot Plot
=========

_thumb: .2, .8
_example_title: Plot distribution.
"""
import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-darkgrid")

data = np.random.normal(0, 1, 1000)
az.plot_dot(data, dotcolor="C1", point_interval=True, figsize=(12, 6))

plt.show()
