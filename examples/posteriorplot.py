"""
Posterior Plot
==============

_thumb: .5, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

trace = az.load_trace('data/non_centered_eight_trace.gzip')
az.posteriorplot(trace, varnames=['theta__0', 'theta__1', 'tau', 'mu'])