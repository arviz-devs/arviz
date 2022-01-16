"""
Posterior Plot
==============

_thumb: .5, .8
_example_title: Plot posterior
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("centered_eight")

coords = {"school": ["Choate"]}
az.plot_posterior(data, var_names=["mu", "theta"], coords=coords, rope=(-1, 1))

plt.show()
