"""
Forest Plot
===========

_thumb: .5, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

centered_data = az.load_arviz_data('centered_eight')
non_centered_data = az.load_arviz_data('non_centered_eight')
fig, axes = az.plot_forest([centered_data, non_centered_data],
                           model_names=['Centered', 'Non Centered'],
                           var_names=['mu'])
axes[0].set_title('Estimated theta for eight schools model')
