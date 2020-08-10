"""
ELPD Plot
=========

_thumb: .6, .5
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

d1 = az.load_arviz_data("centered_eight")
d2 = az.load_arviz_data("non_centered_eight")

az.plot_elpd({"Centered eight": d1, "Non centered eight": d2}, xlabels=True)

plt.show()
