"""
LOO-PIT Overlay Plot
====================

_thumb: .5, .7
_example_title: Plot LOO overlayed on <i>some damn thing</i>.
"""
import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("non_centered_eight")

az.plot_loo_pit(idata=idata, y="obs", color="indigo")

plt.show()
