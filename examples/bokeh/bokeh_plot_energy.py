"""
Energy Plot
===========
"""

import arviz as az

data = az.load_arviz_data("centered_eight")
ax = az.plot_energy(data, figsize=(12, 8), backend="bokeh")
