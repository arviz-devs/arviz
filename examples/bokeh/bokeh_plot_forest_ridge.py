"""
Ridgeplot
=========

_thumb: .8, .5
"""
import arviz as az

rugby_data = az.load_arviz_data("rugby")
ax = az.plot_forest(
    rugby_data,
    kind="ridgeplot",
    var_names=["defs"],
    linewidth=4,
    combined=True,
    ridgeplot_overlap=1.5,
    colors="blue",
    figsize=(9, 4),
    backend="bokeh",
)
