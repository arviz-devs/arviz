"""
Parallel Plot
=============

_thumb: .2, .5
"""
import arviz as az
import numpy as np
import pymc3 as pm

az.style.use('arviz-darkgrid')

data = az.load_arviz_data('centered_eight')
az.parallelplot(data, var_names=['theta', 'tau', 'mu'])
