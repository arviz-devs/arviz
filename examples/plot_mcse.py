"""
Quantile MCSE Plot
=========

_thumb: .3, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

data = az.load_arviz_data('non_centered_eight')
az.plot_mcse(data, var_names=['tau', 'mu'], color="C4", rug=True)
