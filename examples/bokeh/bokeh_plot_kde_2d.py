"""
2D KDE
======
"""

import numpy as np

import arviz as az

az.plot_kde(
    np.random.beta(2, 5, size=100),
    np.random.beta(2, 5, size=100),
    contour_kwargs={"levels": 30},
    contourf_kwargs={"alpha": 0.5, "levels": 30, "cmap": "viridis"},
    backend="bokeh",
)
