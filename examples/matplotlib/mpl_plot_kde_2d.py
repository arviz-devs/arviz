"""
2D KDE
======
_gallery_category: Distributions
"""

import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-doc")

az.plot_kde(
    np.random.beta(2, 5, size=100),
    np.random.beta(2, 5, size=100),
    contour_kwargs={"colors": None, "cmap": plt.cm.viridis, "levels": 30},
    contourf_kwargs={"alpha": 0.5, "levels": 30},
)

plt.show()
