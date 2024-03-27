"""
Quantile MCSE Errobar Plot
==========================
_gallery_category: Inference Diagnostics
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("radon")
az.plot_mcse(data, var_names=["sigma_a"], errorbar=True)

plt.show()
