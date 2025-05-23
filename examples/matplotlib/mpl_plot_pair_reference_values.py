"""
Pairplot with Reference Values
==============================
_gallery_category: Distributions
"""

import matplotlib.pyplot as plt

import arviz as az
import numpy as np

az.style.use("arviz-doc")

data = az.load_arviz_data("centered_eight")

coords = {"school": ["Choate", "Deerfield"]}
reference_values = {
    "mu": 0.0,
    "theta": np.zeros(2),
}

ax = az.plot_pair(
    data,
    var_names=["mu", "theta"],
    kind=["scatter", "kde"],
    kde_kwargs={"fill_last": False},
    coords=coords,
    reference_values=reference_values,
    figsize=(11.5, 5),
)

plt.show()
