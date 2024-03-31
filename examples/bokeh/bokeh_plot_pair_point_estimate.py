"""
Point Estimate Pairplot
=======================
"""

import arviz as az

centered = az.load_arviz_data("centered_eight")

coords = {"school": ["Choate", "Deerfield"]}
ax = az.plot_pair(
    centered,
    var_names=["mu", "theta"],
    kind=["scatter", "kde"],
    kde_kwargs={"fill_last": False},
    marginals=True,
    coords=coords,
    point_estimate="median",
    figsize=(10, 8),
    backend="bokeh",
)
