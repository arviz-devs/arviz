"""
Forest Plot
===========

_thumb: .5, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

trace = az.load_trace('data/centered_eight_trace.gzip')
az.forestplot(trace, varnames=('theta__0', 'theta__1', 'theta__2'))