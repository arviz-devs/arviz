"""
KDE Plot
========

_thumb: .2, .8
"""
import arviz as az
import matplotlib.pyplot as plt

az.style.use('arviz-darkgrid')

trace = az.load_trace('data/non_centered_eight_trace.gzip')

fig, ax = plt.subplots(figsize=(12, 8))
az.kdeplot(trace.tau, fill_alpha=0.1, ax=ax)