"""
Posterior Plot
==============

_thumb: .5, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

non_centered = az.load_arviz_data('non_centered_eight')

az.plot_posterior(non_centered, var_names=("mu", 'theta_tilde',), rope=(-1, 1))
