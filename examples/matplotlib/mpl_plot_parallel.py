"""
Parallel Plot
=============
_gallery_category: Inference Diagnostics
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("centered_eight")
ax = az.plot_parallel(data, var_names=["theta", "tau", "mu"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.show()
