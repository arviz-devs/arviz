"""
Posterior Plot (reducing school dimension)
==========================================

_thumb: .5, .8
"""
import matplotlib.pyplot as plt
import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("centered_eight")

coords = {"school": ["Choate", "Mt. Hermon", "Deerfield"]}
az.plot_posterior(data, var_names=["mu", "theta"], combine_dims={"school"}, coords=coords)

plt.show()
