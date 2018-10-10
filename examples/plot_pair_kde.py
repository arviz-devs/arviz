"""
KDE Pair Plot
=============

_thumb: .2, .5
"""
import arviz as az

az.style.use('arviz-darkgrid')

centered = az.load_arviz_data('centered_eight')

coords = {'school': ['Choate', 'Deerfield']}
az.plot_pair(centered, var_names=['theta', 'mu', 'tau'], kind='kde', coords=coords,
             divergences=True, textsize=22)
