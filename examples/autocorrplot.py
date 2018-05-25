"""
Autocorrelation Plot
====================

_thumb: .8, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

trace = az.load_trace('data/centered_eight_trace.gzip')
az.autocorrplot(trace, varnames=('tau', 'theta__0', 'mu'))