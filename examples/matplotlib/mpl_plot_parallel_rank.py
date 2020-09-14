"""
Rank Parallel Plot
==================

_thumb: .2, .5
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("centered_eight")
ax = az.plot_parallel(data, var_names=["theta", "tau", "mu"], norm_method="rank")
ax.set_xticklabels(ax.get_xticklabels(), rotation=70)

plt.show()
