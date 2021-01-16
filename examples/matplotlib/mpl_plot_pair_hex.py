"""
Hexbin PairPlot
===============

_thumb: .2, .5
_example_title: Pair plot on hex grid
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

centered = az.load_arviz_data("centered_eight")

coords = {"school": ["Choate", "Deerfield"]}
az.plot_pair(
    centered,
    var_names=["theta", "mu", "tau"],
    kind="hexbin",
    coords=coords,
    colorbar=True,
    divergences=True,
)
plt.show()
