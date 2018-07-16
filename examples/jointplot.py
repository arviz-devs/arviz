"""
Joint Plot
==========

_thumb: .5, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')


data = az.load_data('data/non_centered_eight.nc')

az.jointplot(data,
             var_names=['theta'],
             coords={'school': ['Choate', 'Phillips Andover']},
             kind='hexbin',
             figsize=(10, 10))
