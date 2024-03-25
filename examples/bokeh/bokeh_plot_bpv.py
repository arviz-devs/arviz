"""
Bayesian u-value Plot
=====================
"""

import arviz as az

data = az.load_arviz_data("regression1d")
ax = az.plot_bpv(data, backend="bokeh")
