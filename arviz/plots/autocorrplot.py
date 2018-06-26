import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import _scale_text, default_grid, selection_to_string, xarray_var_iter
from ..utils import convert_to_xarray


def autocorrplot(posterior, var_names=None, max_lag=100, symmetric_plot=False, combined=False,
                 figsize=None, textsize=None, skip_first=0):
    """
    Bar plot of the autocorrelation function for a posterior.

    Parameters
    ----------
    posterior : xarray, or object that can be converted (pystan or pymc3 draws)
        Posterior samples
    var_names : list of variable names, optional
        Variables to be plotted, if None all variable are plotted.
        Vector-value stochastics are handled automatically.
    max_lag : int, optional
        Maximum lag to calculate autocorrelation. Defaults to 100.
    symmetric_plot : boolean, optional
        Plot from either [0, +lag] or [-lag, lag]. Defaults to False, [-, +lag].
    combined : bool
        Flag for combining multiple chains into a single chain. If False (default), chains will be
        plotted separately.
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inches.
        Note this is not used if ax is supplied.
    textsize: int
        Text size for labels, titles and lines. If None it will be autoscaled based on figsize.
    skip_first : int, optional
        Number of first samples not shown in plots (burn-in).

    Returns
    -------
    ax : matplotlib axes
    """
    data = convert_to_xarray(posterior)
    data = data.where(data.draw >= skip_first).dropna('draw')

    if symmetric_plot:
        min_lag = -max_lag
    else:
        min_lag = 0

    plotters = list(xarray_var_iter(data, var_names, combined))
    rows, cols = default_grid(len(plotters))

    if figsize is None:
        figsize = (3 * cols, 2.5 * rows)
    textsize, linewidth, _ = _scale_text(figsize, textsize, 1)

    _, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, sharex=True, sharey=True)

    axes = np.atleast_2d(axes)  # in case of only 1 plot
    y_min = 0
    ax = None
    for (var_name, selection, x), ax in zip(plotters, axes.flatten()):
        y = x - x.mean()
        y = np.correlate(y, y, mode=2)
        y = y / np.abs(y).max()
        midpoint = len(y) // 2
        ax.vlines(x=np.arange(min_lag, max_lag),
                  ymin=np.zeros(max_lag - min_lag),
                  ymax=y[midpoint + min_lag:midpoint + max_lag],
                  lw=linewidth)
        ax.hlines(0, min_lag, max_lag, 'steelblue')
        ax.set_title('{} ({})'.format(var_name, selection_to_string(selection)), fontsize=textsize)
        ax.tick_params(labelsize=textsize)
        y_min = min(y_min, y.min())

    if ax is not None:
        ax.set_xlim(min_lag, max_lag)
        ax.set_ylim(y_min, 1)
    return ax
