"""
Quantile MCSE Errobar Plot
==========================

_thumb: .3, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

data = az.load_arviz_data('radon')
az.plot_mcse(data, var_names=["b"], color="C4", errorbar=True)
