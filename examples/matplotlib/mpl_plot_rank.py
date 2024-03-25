"""
Rank plot
=========
_gallery_category: Inference Diagnostics
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

fig, axes = plt.subplots(1, 2)

data = az.load_arviz_data("centered_eight")
az.plot_rank(data, var_names=("tau", "mu"), ax=axes)

fig.suptitle("Rank (All Chains)")

plt.show()
