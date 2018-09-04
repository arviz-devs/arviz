"""Parallel coordinates plot showing posterior points with and without divergences marked."""
import matplotlib.pyplot as plt
import numpy as np

from ..utils import trace_to_dataframe, get_varnames, get_stats
from .plot_utils import _scale_text


def parallelplot(trace, varnames=None, figsize=None, textsize=None, legend=True, colornd='k',
                 colord='C1', shadend=.025, skip_first=0, ax=None):
    """Parallel coordinates plot showing posterior points with and without divergences marked.

    Described by https://arxiv.org/abs/1709.01449, suggested by Ari Hartikainen

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted. Can be used to change the order
        of the plotted variables
    figsize : figure size tuple
        If None, size is (12 x 6)
    textsize: int
        Text size for labels. If None it will be autoscaled based on figsize.
    legend : bool
        Flag for plotting legend (defaults to True)
    colornd : valid matplotlib color
        color for non-divergent points. Defaults to 'k'
    colord : valid matplotlib color
        color for divergent points. Defaults to 'C1'
    shadend : float
        Alpha blending value for non-divergent points, between 0 (invisible) and 1 (opaque).
        Defaults to .025
    skip_first : int, optional
        Number of first samples not shown in plots (burn-in).
    ax : axes
        Matplotlib axes.

    Returns
    -------
    ax : matplotlib axes
    """
    divergent = get_stats(trace[skip_first:], 'diverging')
    trace = trace_to_dataframe(trace[skip_first:])
    varnames = get_varnames(trace, varnames)

    if len(varnames) < 2:
        raise ValueError('This plot needs at least two variables')

    trace = trace[varnames]

    if figsize is None:
        figsize = (12, 6)

    if textsize is None:
        textsize, _, _ = _scale_text(figsize, textsize=textsize, scale_ratio=1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(trace.values[divergent == 0].T, color=colornd, alpha=shadend)
    if np.any(divergent):
        ax.plot(trace.values[divergent == 1].T, color=colord, lw=1)

    ax.tick_params(labelsize=textsize)
    ax.set_xticks(range(trace.shape[1]))
    ax.set_xticklabels(varnames)

    if legend:
        ax.plot([], color=colornd, label='non-divergent')
        if np.any(divergent):
            ax.plot([], color=colord, label='divergent')
        ax.legend(fontsize=textsize)

    return ax
