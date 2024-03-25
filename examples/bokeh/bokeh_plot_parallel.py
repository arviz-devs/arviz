"""
Parallel Plot
=============
"""

import arviz as az

data = az.load_arviz_data("centered_eight")
ax = az.plot_parallel(data, var_names=["theta", "tau", "mu"], backend="bokeh")
