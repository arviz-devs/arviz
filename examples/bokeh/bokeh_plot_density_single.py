"""
Density Plot
============
"""

import arviz as az

centered_data = az.load_arviz_data("centered_eight")
ax = az.plot_density(
    [centered_data],
    data_labels=["Centered"],
    var_names=["theta"],
    shade=0.1,
    backend="bokeh",
)
