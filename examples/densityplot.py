"""
Density Plot
============

_thumb: .5, .5
"""
import arviz as az

trace = az.load_trace('data/centered_eight_trace.gzip')
az.densityplot(trace, varnames=('tau', 'theta__0'))