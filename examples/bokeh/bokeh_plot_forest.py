"""
Forest Plot
===========

_thumb: .5, .8
"""
import arviz as az

centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")
ax = az.plot_forest(
    [centered_data, non_centered_data],
    model_names=["Centered", "Non Centered"],
    var_names=["mu"],
    backend="bokeh",
)
