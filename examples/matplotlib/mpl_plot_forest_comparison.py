"""
Forest Plot Comparison
======================
_gallery_category: Distribution Comparison
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")
ax = az.plot_forest(
    [centered_data, non_centered_data],
    model_names=["Centered", "Non Centered"],
    var_names=["mu"],
    figsize=(11.5, 5),
)
ax[0].set_title("Estimated theta for eight schools model")

plt.show()
