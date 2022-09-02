"""
Traceplot rank_bars
===================
_gallery_category: Mixed
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("non_centered_eight")
az.plot_trace(data, var_names=("tau", "mu"), kind="rank_bars")

plt.show()
