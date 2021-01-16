"""
Joint Plot
==========

_thumb: .5, .8
_example_title: Plot joint distribution
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")


data = az.load_arviz_data("non_centered_eight")

az.plot_pair(
    data,
    var_names=["theta"],
    coords={"school": ["Choate", "Phillips Andover"]},
    kind="hexbin",
    marginals=True,
    figsize=(10, 10),
)
plt.show()
