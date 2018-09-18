"""
Joint Plot
==========

_thumb: .5, .8
"""
import arviz as az

az.style.use('arviz-darkgrid')


data = az.load_arviz_data('non_centered_eight')

az.plot_joint(data,
              var_names=['theta'],
              coords={'school': ['Choate', 'Phillips Andover']},
              kind='hexbin',
              figsize=(10, 10))
