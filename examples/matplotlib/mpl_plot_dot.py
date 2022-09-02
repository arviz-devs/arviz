"""
Dot Plot
=========
"""
import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-doc")

data = np.random.normal(0, 1, 1000)
az.plot_dot(data, dotcolor="C1", point_interval=True, figsize=(12, 6))

plt.show()
