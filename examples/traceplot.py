"""
Traceplot
=========

_thumb: .8, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

data = az.load_data('data/non_centered_eight.nc')
az.traceplot(data, var_names=('tau', 'mu'))
