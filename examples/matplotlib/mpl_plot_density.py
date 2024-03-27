"""
Density Plot (Comparison)
=========================
_gallery_category: Distribution Comparison
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")

axes = az.plot_density(
    [centered_data, non_centered_data],
    data_labels=["Centered", "Non-Centered"],
    var_names=["theta"],
    shade=0.2,
)

fig = axes.flatten()[0].get_figure()
fig.suptitle("94% High Density Intervals for Theta")

plt.show()
