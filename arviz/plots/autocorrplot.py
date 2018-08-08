import numpy as np

from .plot_utils import _scale_text, default_grid, make_label, xarray_var_iter, _create_axes_grid
from ..utils import convert_to_xarray
from ..stats.diagnostics import autocorr


def autocorrplot(posterior, var_names=None, max_lag=100, combined=False,
                 figsize=None, textsize=None):
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
    combined : bool
        Flag for combining multiple chains into a single chain. If False (default), chains will be
        plotted separately.
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inches.
        Note this is not used if ax is supplied.
    textsize: int
        Text size for labels, titles and lines. If None it will be autoscaled based on figsize.

    Returns
    -------
    axes : matplotlib axes
    """
    data = convert_to_xarray(posterior)

    plotters = list(xarray_var_iter(data, var_names, combined))
    length_plotters = len(plotters)
    rows, cols = default_grid(length_plotters)

    if figsize is None:
        figsize = (3 * cols, 2.5 * rows)
    textsize, linewidth, _ = _scale_text(figsize, textsize, 1.5)


    _, axes = _create_axes_grid(length_plotters, rows, cols, figsize=figsize,
                                squeeze=False, sharex=True, sharey=True)

    axes = np.atleast_2d(axes)  # in case of only 1 plot
    for (var_name, selection, x), ax in zip(plotters, axes.flatten()):
        x_prime = x

        if combined:
            x_prime = x.flatten()

        y = autocorr(x_prime)

        ax.vlines(x=np.arange(0, max_lag),
                  ymin=0, ymax=y[0:max_lag],
                  lw=linewidth)
        ax.hlines(0, 0, max_lag, 'steelblue')
        ax.set_title(make_label(var_name, selection), fontsize=textsize)
        ax.tick_params(labelsize=textsize)

    if axes.size > 0:
        axes[0, 0].set_xlim(0, max_lag)
        axes[0, 0].set_ylim(-1, 1)

    return axes
