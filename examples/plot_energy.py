"""
Energy Plot
===========

_thumb: .7, .5
"""
import arviz as az
import matplotlib.pyplot as plt

# close all the figures, if open from previous commands
plt.close("all")

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("centered_eight")
az.plot_energy(data, figsize=(12, 8))
