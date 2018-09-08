"""
Violinplot
==========

_thumb: .2, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')

non_centered = az.load_arviz_data('non_centered_eight')
az.violintraceplot(non_centered, var_names=["mu", "tau"], textsize=8)
