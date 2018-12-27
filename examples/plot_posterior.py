"""
Posterior Plot
==============

_thumb: .5, .8
"""
import arviz as az
import matplotlib.pyplot as plt

# close all the figures, if open from previous commands
plt.close("all")

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("centered_eight")

coords = {"school": ["Choate"]}
az.plot_posterior(data, var_names=["mu", "theta"], coords=coords, rope=(-1, 1))
