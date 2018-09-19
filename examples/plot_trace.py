"""
Traceplot
=========

_thumb: .1, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

data = az.load_arviz_data('non_centered_eight')
az.plot_trace(data, var_names=('tau', 'mu'))
