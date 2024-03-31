"""
KDE Plot
========
_gallery_category: Distributions
"""

import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("centered_eight")

# Combine posterior draws for from xarray of (4,500) to ndarray (2000,)
y_hat = np.concatenate(data.posterior_predictive["obs"].values)

az.plot_kde(
    y_hat,
    label="Estimated Effect\n of SAT Prep",
    rug=True,
    plot_kwargs={"linewidth": 2},
    rug_kwargs={"alpha": 0.05},
)

plt.show()
