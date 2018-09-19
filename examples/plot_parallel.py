"""
Parallel Plot
=============

_thumb: .2, .5
"""
import arviz as az

az.style.use('arviz-darkgrid')

data = az.load_arviz_data('centered_eight')
az.plot_parallel(data, var_names=['theta', 'tau', 'mu'])
