"""
Hexbin PairPlot
===============

_thumb: .2, .5
"""
import arviz as az

az.style.use('arviz-darkgrid')

centered = az.load_arviz_data('centered_eight')

az.pairplot(centered, var_names=['theta', "mu"], kind='hexbin', colorbar=True, divergences=True)
