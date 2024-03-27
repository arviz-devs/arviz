"""
Forest Plot
===========
_gallery_category: Distributions
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

centered_data = az.load_arviz_data("centered_eight")
ax = az.plot_forest(
    centered_data,
    var_names=["~tau"],
    combined=False,
    figsize=(11.5, 5),
    colors="C1",
)

plt.show()
