"""
Posterior Predictive Check Cumulative Plot
==========================================
"""

import arviz as az

data = az.load_arviz_data("non_centered_eight")
ax = az.plot_ppc(data, alpha=0.03, kind="cumulative", figsize=(12, 6), backend="bokeh")
