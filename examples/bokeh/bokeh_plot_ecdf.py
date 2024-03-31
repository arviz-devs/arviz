"""
ECDF Plot
=========
"""

from scipy.stats import norm

import arviz as az

sample = norm(0, 1).rvs(1000)
distribution = norm(0, 1)

az.plot_ecdf(sample, cdf=distribution.cdf, confidence_bands=True, backend="bokeh")
