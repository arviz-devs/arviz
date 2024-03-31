"""
Forest Plot
===========
"""

import arviz as az

centered_data = az.load_arviz_data("centered_eight")
ax = az.plot_forest(
    centered_data,
    var_names=["~tau"],
    combined=False,
    figsize=(11.5, 5),
    backend="bokeh",
)
