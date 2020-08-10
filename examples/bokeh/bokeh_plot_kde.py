"""
KDE Plot Bokeh
==============

_thumb: .2, .8
"""
import bokeh.plotting as bkp
import numpy as np

import arviz as az

data = az.load_arviz_data("centered_eight")

# Combine posterior draws for from xarray of (4,500) to ndarray (2000,)
y_hat = np.concatenate(data.posterior_predictive["obs"].values)

figure_kwargs = dict(height=500, width=500, output_backend="webgl")
ax = bkp.figure(**figure_kwargs)

ax = az.plot_kde(
    y_hat,
    label="Estimated Effect\n of SAT Prep",
    rug=True,
    plot_kwargs={"line_width": 2, "line_color": "black"},
    rug_kwargs={"line_color": "black"},
    backend="bokeh",
    ax=ax,
)
