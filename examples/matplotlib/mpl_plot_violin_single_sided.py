"""
Single-Sided Violin Plot
========================
_gallery_category: Distribution Comparison
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("rugby")

labeller = az.labels.MapLabeller(var_name_map={"defs": "atts | defs"})
axes = az.plot_violin(
    data,
    var_names=["atts"],
    side="left",
    show=False,
    figsize=(11.5, 5),
)
az.plot_violin(
    data,
    var_names=["defs"],
    side="right",
    labeller=labeller,
    ax=axes,
    show=True,
)

fig = axes.flatten()[0].get_figure()
fig.suptitle("Attack/Defense of Rugby Teams")
fig.tight_layout()

plt.show()
