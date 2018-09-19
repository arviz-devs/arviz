"""Pareto tail indices plot."""
import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import _scale_text


def plot_khat(khats, figsize=None, textsize=None, ax=None, hlines_kwargs=None, **kwargs):
    R"""
    Plot Pareto tail indices.

    Parameters
    ----------
    khats : array
      Pareto tail indices.
    figsize : figure size tuple
      If None, size is (8, 4)
    textsize: int
      Text size for labels. If None it will be autoscaled based on figsize.
    ax: axes
      Matplotlib axes
    hlines_kwargs: dictionary
      Additional keywords passed to ax.hlines
    kwargs :
      Additional keywords passed to ax.scatter

    Returns
    -------
    ax : axes
      Matplotlib axes.
    """
    if figsize is None:
        figsize = (8, 5)

    if hlines_kwargs is None:
        hlines_kwargs = {}

    textsize, linewidth, markersize = _scale_text(figsize, textsize=textsize)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.hlines([0, .5, .7, 1], xmin=-1, xmax=len(khats)+1,
              alpha=.25, linewidth=linewidth, **hlines_kwargs)

    alphas = .5 + .5*(khats > .5)
    rgba_c = np.zeros((len(khats), 4))
    rgba_c[:, 2] = .8
    rgba_c[:, 3] = alphas
    ax.scatter(np.arange(len(khats)), khats, c=rgba_c, marker='+', markersize=markersize, **kwargs)
    ax.set_xlabel('Data point', fontsize=textsize)
    ax.set_ylabel(r'Shape parameter κ', fontsize=textsize)
    ax.tick_params(labelsize=textsize)
    return ax
