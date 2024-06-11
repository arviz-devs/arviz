"""
ECDF Plot
=========
_gallery_category: Distributions
"""

import warnings
import matplotlib.pyplot as plt
from scipy.stats import norm

import arviz as az

az.style.use("arviz-doc")

sample = norm(0, 1).rvs(1000)
distribution = norm(0, 1)

warnings.filterwarnings("ignore", category=az.utils.BehaviourChangeWarning)

az.plot_ecdf(sample, cdf=distribution.cdf, confidence_bands=True)

plt.show()
