"""
Posterior Plot
==============

_thumb: .5, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

non_centered = az.load_arviz_data('non_centered_eight')

az.posteriorplot(non_centered, var_names=('theta_tilde',), ref_val=0, rope=(-1, 1), textsize=11)
