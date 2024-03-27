"""
Dist Plot
=========
_gallery_category: Distributions
"""

import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-doc")

data_poisson = np.random.poisson(4, 1000)
data_gaussian = np.random.normal(0, 1, 1000)

fig, ax = plt.subplots(1, 2)
fig.suptitle("Distributions")

ax[0].set_title("Poisson")
az.plot_dist(data_poisson, color="C1", label="Poisson", ax=ax[0])

ax[1].set_title("Gaussian")
az.plot_dist(data_gaussian, color="C2", label="Gaussian", ax=ax[1])

plt.show()
