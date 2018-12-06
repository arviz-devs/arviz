"""Autocorrelation plot of data."""
import numpy as np

from ..data import convert_to_dataset
from ..stats.diagnostics import autocorr
from .plot_utils import (
    _scale_fig_size,
    default_grid,
    make_label,
    xarray_var_iter,
    _create_axes_grid,
)
from ..utils import _var_names


def plot_autocorr(data, var_names=None, max_lag=100, combined=False, figsize=None, textsize=None):
    """Bar plot of the autocorrelation function for a sequence of data.

    Useful in particular for posteriors from MCMC samples which may display correlation.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names, optional
        Variables to be plotted, if None all variable are plotted.
        Vector-value stochastics are handled automatically.
    max_lag : int, optional
        Maximum lag to calculate autocorrelation. Defaults to 100.
    combined : bool
        Flag for combining multiple chains into a single chain. If False (default), chains will be
        plotted separately.
    figsize : tuple
        Figure size. If None it will be defined automatically.
        Note this is not used if ax is supplied.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.

    Returns
    -------
    axes : matplotlib axes

    Examples
    --------
    Plot default autocorrelation

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_autocorr(data)

    Plot subset variables by specifying variable name exactly

    .. plot::
        :context: close-figs

        >>> az.plot_autocorr(data, var_names=['mu', 'tau'] )


    Combine chains collapsing by variable

    .. plot::
        :context: close-figs

        >>> az.plot_autocorr(data, var_names=['mu', 'tau'], combined=True)


    Specify maximum lag (x axis bound)

    .. plot::
        :context: close-figs

        >>> az.plot_autocorr(data, var_names=['mu', 'tau'], max_lag=200, combined=True)
    """
    data = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names)

    plotters = list(xarray_var_iter(data, var_names, combined))
    length_plotters = len(plotters)
    rows, cols = default_grid(length_plotters)

    figsize, _, titlesize, xt_labelsize, linewidth, _ = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    _, axes = _create_axes_grid(
        length_plotters, rows, cols, figsize=figsize, squeeze=False, sharex=True, sharey=True
    )

    axes = np.atleast_2d(axes)  # in case of only 1 plot
    for (var_name, selection, x), ax in zip(plotters, axes.flatten()):
        x_prime = x

        if combined:
            x_prime = x.flatten()

        y = autocorr(x_prime)

        ax.vlines(x=np.arange(0, max_lag), ymin=0, ymax=y[0:max_lag], lw=linewidth)
        ax.hlines(0, 0, max_lag, "steelblue")
        ax.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
        ax.tick_params(labelsize=xt_labelsize)

    if axes.size > 0:
        axes[0, 0].set_xlim(0, max_lag)
        axes[0, 0].set_ylim(-1, 1)

    return axes
