"""
Plot HDI
========

_thumb: .8, .8
"""
import bokeh.plotting as bkp
import numpy as np

import arviz as az

x_data = np.random.normal(0, 1, 100)
y_data = 2 + x_data * 0.5
y_data_rep = np.random.normal(y_data, 0.5, (4, 200, 100))
x_data_sorted = np.sort(x_data)

ax = az.plot_hdi(x_data, y_data_rep, color="red", backend="bokeh", show=False)
ax.line(x_data_sorted, 2 + x_data_sorted * 0.5, line_color="black", line_width=3)

if az.rcParams["plot.bokeh.show"]:
    bkp.show(ax)
