"""
Plot HPD
========

_thumb: .8, .8
"""
import numpy as np
import arviz as az

x_data = np.random.normal(0, 1, 100)
y_data = 2 + x_data * 0.5
y_data_rep = np.random.normal(y_data, 0.5, (200, 100))

ax = az.plot_hpd(x_data, y_data_rep, color="red", backend="bokeh")
ax.line(x_data, y_data, "black")
