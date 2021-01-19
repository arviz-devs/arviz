"""
Ridgeplot
=========

_thumb: .8, .5
_example_title: Forest plot with individual ridges
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

rugby_data = az.load_arviz_data("rugby")
axes = az.plot_forest(
    rugby_data,
    kind="ridgeplot",
    var_names=["defs"],
    linewidth=4,
    combined=True,
    ridgeplot_overlap=1.5,
    colors="blue",
    figsize=(9, 4),
)
axes[0].set_title("Relative defensive strength\nof Six Nation rugby teams")

plt.show()
