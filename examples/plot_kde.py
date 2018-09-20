"""
KDE Plot
========

_thumb: .2, .8
"""
import arviz as az
import numpy as np

az.style.use('arviz-darkgrid')

data = az.load_arviz_data('centered_eight')

# Combine posterior draws for from xarray of (4,500) to ndarray (2000,)
y_hat = np.concatenate(data.posterior_predictive["obs"].values)

ax = az.plot_kde(y_hat, label='Estimated Effect of SAT Prep', rug=True,
                 plot_kwargs={'linewidth': 5, 'color': 'black'},
                 rug_kwargs={'color': 'black'})
