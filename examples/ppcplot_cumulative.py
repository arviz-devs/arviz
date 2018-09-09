"""
Posterior Predictive Check Cumulative Plot
==========================================

_thumb: .6, .5
"""
import arviz as az

az.style.use('arviz-darkgrid')

data = az.load_arviz_data('non_centered_eight')
az.ppcplot(data, alpha=0.03, kind='cumulative', figsize=(12, 6), textsize=14)