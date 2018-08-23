"""
Autocorrelation Plot
====================

_thumb: .8, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

data = az.load_arviz_data('centered_eight')
az.autocorrplot(data, var_names=('tau', 'mu'))
