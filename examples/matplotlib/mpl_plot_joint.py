"""
Joint Plot
==========
_gallery_category: Distributions
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("non_centered_eight")

az.plot_pair(
    data,
    var_names=["theta"],
    coords={"school": ["Choate", "Phillips Andover"]},
    kind="hexbin",
    marginals=True,
    figsize=(11.5, 11.5),
)
plt.show()
