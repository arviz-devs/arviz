"""
Compare Plot
============

_thumb: .5, .5
"""
import arviz as az

model_compare = az.compare(
    {
        "Centered 8 schools": az.load_arviz_data("centered_eight"),
        "Non-centered 8 schools": az.load_arviz_data("non_centered_eight"),
    }
)
ax = az.plot_compare(model_compare, figsize=(12, 4), backend="bokeh")
