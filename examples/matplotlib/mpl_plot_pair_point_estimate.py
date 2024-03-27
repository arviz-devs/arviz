"""
Point Estimate Pairplot
=======================
_gallery_category: Distributions
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("centered_eight")

coords = {"school": ["Choate", "Deerfield"]}
ax = az.plot_pair(
    data,
    var_names=["mu", "theta"],
    kind=["scatter", "kde"],
    kde_kwargs={"fill_last": False},
    marginals=True,
    coords=coords,
    point_estimate="median",
    figsize=(11.5, 5),
)

plt.show()
