"""
ECDF Plot (Comparison)
======================
"""

import matplotlib.pyplot as plt
from scipy.stats import norm, ecdf

import arviz as az

az.style.use("arviz-doc")

sample1 = norm(0, 1).rvs(1000, random_state=523)
sample2 = norm(0.1, 1).rvs(1000, random_state=74)

az.plot_ecdf(
    sample1,
    cdf=ecdf(sample2).cdf.evaluate,
    difference=True,
    confidence_bands=True,
    ci_prob=0.9,
    backend="bokeh",
)
