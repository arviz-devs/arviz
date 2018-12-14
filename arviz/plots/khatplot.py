"""Pareto tail indices plot."""
import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import _scale_fig_size


def plot_khat(
    khats, figsize=None, textsize=None, markersize=None, ax=None, hlines_kwargs=None, **kwargs
):
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
        markersize for scatter plot. Defaults to `None` in which case it will
        be chosen based on autoscaling for figsize.
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

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, scaled_markersize) = _scale_fig_size(
        figsize, textsize
    )

    if markersize is None:
        markersize = scaled_markersize

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    ax.hlines(
        [0, 0.5, 0.7, 1],
        xmin=-1,
        xmax=len(khats) + 1,
        alpha=0.25,
        linewidth=linewidth,
        **hlines_kwargs
    )

    alphas = 0.5 + 0.5 * (khats > 0.5)
    rgba_c = np.zeros((len(khats), 4))
    rgba_c[:, 2] = 0.8
    rgba_c[:, 3] = alphas
    ax.scatter(np.arange(len(khats)), khats, c=rgba_c, marker="+", s=markersize, **kwargs)
    ax.set_xlabel("Data point", fontsize=ax_labelsize)
    ax.set_ylabel(r"Shape parameter κ", fontsize=ax_labelsize)
    ax.tick_params(labelsize=xt_labelsize)
    return ax
