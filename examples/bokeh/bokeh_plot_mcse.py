"""
Quantile Monte Carlo Standard Error Plot
========================================

_thumb: .5, .8
"""
import arviz as az

data = az.load_arviz_data("centered_eight")
az.plot_mcse(data, var_names=["tau", "mu"], rug=True, extra_methods=True, backend="bokeh")
