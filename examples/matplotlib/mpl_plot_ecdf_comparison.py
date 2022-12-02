"""
ECDF Plot (Comparison)
======================
_gallery_category: Distribution Comparison
"""
import matplotlib.pyplot as plt
from scipy.stats import norm

import arviz as az

az.style.use("arviz-doc")

sample1 = norm(0, 1).rvs(1000, random_state=523)
sample2 = norm(0.1, 1).rvs(1000, random_state=74)

az.plot_ecdf(sample1, sample2, difference=True, confidence_bands=True, fpr=.1)

plt.show()
