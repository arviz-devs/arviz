"""
KDE Plot
========

_thumb: .2, .8
"""
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

az.style.use('arviz-darkgrid')

ax = az.plot_kde(np.random.gumbel(size=100), label='100 gumbel samples', rug=True,
                 plot_kwargs={'linewidth': 5, 'color': 'black'},
                 rug_kwargs={'color': 'black'})
