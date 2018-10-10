"""
Ridgeplot
=========

_thumb: .8, .5
"""
import arviz as az

az.style.use('arviz-darkgrid')

non_centered_data = az.load_arviz_data('non_centered_eight')
fig, axes = az.plot_forest(non_centered_data,
                           kind='ridgeplot',
                           var_names=['theta'],
                           combined=True,
                           ridgeplot_overlap=3,
                           colors='white',
                           figsize=(9, 7))
axes[0].set_title('Estimated theta for 8 schools model')
