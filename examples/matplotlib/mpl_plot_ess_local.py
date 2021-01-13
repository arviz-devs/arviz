"""
ESS Local Plot
==============

_thumb: .6, .5
_example_title: Plot local ESS
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("non_centered_eight")

az.plot_ess(idata, var_names=["mu"], kind="local", marker="_", ms=20, mew=2, rug=True)

plt.show()
