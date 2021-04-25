"""
2d KDE with HDI Contours
========================

_thumb: .5, .8
_example_title: 2d KDE with HDI Contours
"""
import numpy as np

import arviz as az

az.style.use("arviz-darkgrid")

rng = np.random.default_rng()
data = rng.multivariate_normal([2, 2], [[1, 0.4], [0.4, 0.8]], 1000000)

az.plot_kde(
    data[:, 0],
    data[:, 1],
    hdi_probs=[0.393, 0.865, 0.989],  # 1, 2 and 3 sigma contours
    contourf_kwargs={"cmap": "Blues"},
    backend="bokeh",
)
