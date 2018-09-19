"""
Energy Plot
===========

_thumb: .7, .5
"""
import arviz as az
import numpy as np
import pymc3 as pm

az.style.use('arviz-darkgrid')

data = az.load_arviz_data('centered_eight')
az.energyplot(data, figsize=(12, 8))
