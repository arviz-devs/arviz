"""Pareto tail indices plot."""
import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import _scale_fig_size


def plot_khat(khats, figsize=None, textsize=None, markersize=5, ax=None,
              hlines_kwargs=None, **kwargs):
    """
    Plot Pareto tail indices.

    Parameters
    ----------
    khats : array
        Pareto tail indices.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    markersize: int
        markersize for scatter plot. Defaults to 5
    ax: axes, opt
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
    if hlines_kwargs is None:
        hlines_kwargs = {}

    (figsize, ax_labelsize, _, xt_labelsize,
     linewidth, markersize) = _scale_fig_size(figsize, textsize)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.hlines([0, .5, .7, 1], xmin=-1, xmax=len(khats)+1,
              alpha=.25, linewidth=linewidth, **hlines_kwargs)

    alphas = .5 + .5*(khats > .5)
    rgba_c = np.zeros((len(khats), 4))
    rgba_c[:, 2] = .8
    rgba_c[:, 3] = alphas
    ax.scatter(np.arange(len(khats)), khats, c=rgba_c, marker='+', s=markersize, **kwargs)
    ax.set_xlabel('Data point', fontsize=ax_labelsize)
    ax.set_ylabel(r'Shape parameter κ', fontsize=ax_labelsize)
    ax.tick_params(labelsize=xt_labelsize)
    return ax
