"""
Joint Plot
==========

_thumb: .5, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

trace = az.load_trace('data/non_centered_eight_trace.gzip')
az.jointplot(trace, kind='hexbin', varnames=('tau', 'mu'))