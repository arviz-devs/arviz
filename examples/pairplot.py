"""
Pair Plot
=========

_thumb: .2, .5
"""
import arviz as az

az.style.use('arviz-darkgrid')

centered = az.load_arviz_data('centered_eight')

coords = {'school': ["Choate", "Deerfield"]}
az.pairplot(centered, var_names=['theta', "mu"], coords=None, divergences=True)


