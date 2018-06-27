"""
Density Plot
============

_thumb: .5, .5
"""
import arviz as az

az.style.use('arviz-darkgrid')

centered_data = az.load_data('data/centered_eight.nc')
non_centered_data = az.load_data('data/non_centered_eight.nc')
az.densityplot([centered_data, non_centered_data], ['Centered', 'Non Centered'],
                var_names=['theta'], shade=0.1, alpha=0.01)
