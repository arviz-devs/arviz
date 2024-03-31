"""
Forest Plot with ESS
====================
"""

import arviz as az

centered_data = az.load_arviz_data("centered_eight")
ax = az.plot_forest(
    centered_data,
    var_names=["theta"],
    figsize=(11.5, 5),
    ess=True,
    # r_hat=True,
    backend="bokeh",
)
