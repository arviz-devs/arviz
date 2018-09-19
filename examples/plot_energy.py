"""
Energy Plot
===========

_thumb: .7, .5
"""
import arviz as az

az.style.use('arviz-darkgrid')

data = az.load_arviz_data('centered_eight')
az.plot_energy(data, figsize=(12, 8))
