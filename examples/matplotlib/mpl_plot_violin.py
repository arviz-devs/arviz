"""
Violin plot
===========
_gallery_category: Distributions
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

data = az.load_arviz_data("non_centered_eight")
az.plot_violin(data, var_names=["mu", "tau"], figsize=(11.5, 5))

plt.show()
