"""
ESS Evolution Plot
==================
_gallery_category: Inference Diagnostics
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

idata = az.load_arviz_data("radon")

az.plot_ess(idata, var_names=["b"], kind="evolution")

plt.show()
