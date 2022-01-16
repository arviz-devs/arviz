"""
Separation Plot
===============

_thumb: .5, .5
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("classification10d")

az.plot_separation(idata=idata, y="outcome", y_hat="outcome", figsize=(8, 1))

plt.show()
