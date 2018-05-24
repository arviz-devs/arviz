"""
Traceplot
=========

_thumb: .8, .8
"""
import arviz as az

trace = az.load_trace('data/centered_eight_trace.gzip')
az.traceplot(trace, varnames=('tau', 'theta__0'))