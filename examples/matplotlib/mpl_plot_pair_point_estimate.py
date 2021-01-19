"""
Point Estimate Pairplot
=======================

_thumb: .2, .5
_example_title: Pair plot with point estimate markings
"""
import matplotlib.pyplot as plt

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
)

plt.show()
