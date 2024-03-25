"""
Dot Plot
=========
_gallery_category: Distributions
"""

import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-doc")

data = np.random.normal(0, 1, 1000)
ax = az.plot_dot(data, dotcolor="C1", point_interval=True)

ax.set_title("Gaussian Distribution")

plt.show()
