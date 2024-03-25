"""
Posterior Predictive Check Plot
===============================
_gallery_category: Model Checking
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("non_centered_eight")
az.plot_ppc(data, data_pairs={"obs": "obs"}, alpha=0.03, textsize=14)

plt.show()
