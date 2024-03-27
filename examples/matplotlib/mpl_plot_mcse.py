"""
Quantile Monte Carlo Standard Error Plot
========================================
_gallery_category: Inference Diagnostics
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("centered_eight")
az.plot_mcse(data, var_names=["tau", "mu"], rug=True, extra_methods=True)

plt.show()
