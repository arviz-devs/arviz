"""
Posterior Plot (reducing school dimension)
==========================================

_thumb: .5, .8
"""
import arviz as az

data = az.load_arviz_data("centered_eight")

coords = {"school": ["Choate", "Mt. Hermon", "Deerfield"]}
ax = az.plot_posterior(
    data,
    var_names=["mu", "theta"],
    combine_dims={"school"},
    coords=coords,
    backend="bokeh",
)
