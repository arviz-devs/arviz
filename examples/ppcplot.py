"""
Posterior Predictive Check Plot
===============================

_thumb: .6, .5
"""
import arviz as az

az.style.use('arviz-darkgrid')

data = az.load_arviz_data('non_centered_eight')
az.plot_ppc(data, alpha=0.03, figsize=(12, 6), textsize=14)
