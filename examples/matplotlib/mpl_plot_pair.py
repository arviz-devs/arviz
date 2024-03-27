"""
Pair Plot
=========
_gallery_category: Inference Diagnostics
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("centered_eight")

coords = {"school": ["Choate", "Deerfield"]}
az.plot_pair(
    data,
    var_names=["theta", "mu", "tau"],
    coords=coords,
    divergences=True,
    textsize=22,
)

plt.show()
