"""
Dist Plot Bokeh
===============

_thumb: .2, .8
"""
import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import row

import arviz as az

a = np.random.poisson(4, 1000)
b = np.random.normal(0, 1, 1000)

figure_kwargs = dict(height=500, width=500, output_backend="webgl")
ax_poisson = bkp.figure(**figure_kwargs)
ax_normal = bkp.figure(**figure_kwargs)

az.plot_dist(a, color="black", label="Poisson", ax=ax_poisson, backend="bokeh", show=False)
az.plot_dist(b, color="red", label="Gaussian", ax=ax_normal, backend="bokeh", show=False)

ax = row(ax_poisson, ax_normal)

if az.rcParams["plot.bokeh.show"]:
    bkp.show(ax)
