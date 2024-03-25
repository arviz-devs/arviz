"""
LOO-PIT ECDF Plot
=================
_gallery_category: Model Checking
Plot LOO predictive ECDF compared to ECDF of uniform distribution to assess predictive calibration.
"""

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")

idata = az.load_arviz_data("radon")

az.plot_loo_pit(idata, y="y", ecdf=True)

plt.show()
