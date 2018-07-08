"""
Forest Plot
===========

_thumb: .5, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

centered_data = az.load_data('data/centered_eight.nc')
non_centered_data = az.load_data('data/non_centered_eight.nc')
fig, axes = az.forestplot([centered_data, non_centered_data],
                          model_names=['Centered', 'Non Centered'],
                          var_names=['mu'])
axes[0].set_title('Estimated theta for eight schools model')
