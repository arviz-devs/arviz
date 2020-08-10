"""
Posterior Plot
==============

_thumb: .5, .8
"""
import arviz as az

data = az.load_arviz_data("centered_eight")

coords = {"school": ["Choate"]}
ax = az.plot_posterior(
    data, var_names=["mu", "theta"], coords=coords, rope=(-1, 1), backend="bokeh"
)
