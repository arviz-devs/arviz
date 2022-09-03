"""
2d KDE (default style)
======================
_gallery_category: Distributions
"""
import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-doc")

az.plot_kde(np.random.rand(100), np.random.rand(100))

plt.show()
