"""
Traceplot rank_bars
===================

_thumb: .1, .8
_example_title: Trace plot with rank bars
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

data = az.load_arviz_data("non_centered_eight")
az.plot_trace(data, var_names=("tau", "mu"), kind="rank_bars")

plt.show()
