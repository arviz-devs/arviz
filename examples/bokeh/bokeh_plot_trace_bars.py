"""
Traceplot rank_bars Bokeh
=========================
"""

import arviz as az

data = az.load_arviz_data("non_centered_eight")
ax = az.plot_trace(data, var_names=("tau", "mu"), kind="rank_bars", backend="bokeh")
