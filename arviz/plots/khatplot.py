import matplotlib.pyplot as plt
import numpy as np
from .plot_utils import _scale_text


def khatplot(ks, figsize=None, textsize=None, ax=None, hlines_kwargs=None, **kwargs):
    R"""
    Plot Paretto tail indices.

    Parameters
    ----------
    ks : array
      Paretto tail indices.
    figsize : figure size tuple
      If None, size is (10, 6)
    textsize: int
      Text size for labels. If None it will be autoscaled based on figsize.
    ax: axes
      Matplotlib axes
    hlines_kwargs: dictionary
      Aditional keywords passed to ax.hlines

    Returns
    -------
    ax : axes
      Matplotlib axes.
    """

    if figsize is None:
        figsize = (10, 6)

    if hlines_kwargs is None:
        hlines_kwargs = {}

    textsize, linewidth, _ = _scale_text(figsize, textsize=textsize)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.hlines([0, .5, .7, 1], xmin=-1, xmax=len(ks)+1,
              alpha=.25, **hlines_kwargs)

    alphas = .5 + .5*(ks > .5)
    rgba_c = np.zeros((len(ks), 4))
    rgba_c[:, 2] = .8
    rgba_c[:, 3] = alphas
    ax.scatter(np.arange(len(ks)), ks, c=rgba_c, marker='+')
    ax.set_xlabel('Data point')
    ax.set_ylabel(r'Shape parameter $\kappa$')
    return ax
