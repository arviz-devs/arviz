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
                           textsize=11,
                           ridgeplot_overlap=3,
                           colors='white',
                           r_hat=False,
                           n_eff=False)
axes[0].set_title('Estimated theta for eight schools model', fontsize=11)
