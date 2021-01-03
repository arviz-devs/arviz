"""
KDE Plot
========

_thumb: .2, .8
_example_title: Plot Kernel Density Estimation (KDE)
"""
import matplotlib.pyplot as plt
import numpy as np

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("centered_eight")

# Combine posterior draws for from xarray of (4,500) to ndarray (2000,)
y_hat = np.concatenate(data.posterior_predictive["obs"].values)

ax = az.plot_kde(
    y_hat,
    label="Estimated Effect\n of SAT Prep",
    rug=True,
    plot_kwargs={"linewidth": 2, "color": "black"},
    rug_kwargs={"color": "black"},
)
plt.show()
