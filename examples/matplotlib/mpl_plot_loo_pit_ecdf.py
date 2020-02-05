"""
LOO-PIT ECDF Plot
=================

_thumb: .5, .7
"""
import matplotlib.pyplot as plt
import arviz as az

az.style.use("arviz-darkgrid")

idata = az.load_arviz_data("radon")
log_like = idata.sample_stats.log_likelihood.sel(chain=0).values.T
log_weights = az.psislw(-log_like)[0]

az.plot_loo_pit(idata, y="y_like", log_weights=log_weights, ecdf=True, color="maroon")

plt.show()
