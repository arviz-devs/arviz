"""
Bayesian u-value Plot
=====================
_gallery_category: Model Checking
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("regression1d")
az.plot_bpv(data)

plt.show()
