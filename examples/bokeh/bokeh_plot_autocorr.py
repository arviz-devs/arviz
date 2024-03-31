"""
Autocorrelation Plot
====================
"""

import arviz as az

data = az.load_arviz_data("centered_eight")
ax = az.plot_autocorr(data, var_names="tau", backend="bokeh")
