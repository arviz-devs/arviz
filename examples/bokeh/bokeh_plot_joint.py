"""
Joint Plot
==========

_thumb: .5, .8
"""
import arviz as az

data = az.load_arviz_data("non_centered_eight")

ax = az.plot_pair(
    data,
    var_names=["theta"],
    coords={"school": ["Choate", "Phillips Andover"]},
    kind="hexbin",
    figsize=(8, 8),
    marginals=True,
    marginal_kwargs={"plot_kwargs": {"line_width": 3, "line_color": "black"}},
    hexbin_kwargs={"size": 1.5},
    backend="bokeh",
)
