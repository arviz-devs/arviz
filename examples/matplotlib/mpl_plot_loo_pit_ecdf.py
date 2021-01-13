"""
LOO-PIT ECDF Plot
=================

_thumb: .5, .7
_example_title: Plot LOO predictive ECDF compared to ECDF of uniform distribution to assess predictive calibration.
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("radon")

az.plot_loo_pit(idata, y="y", ecdf=True, color="maroon")

plt.show()
