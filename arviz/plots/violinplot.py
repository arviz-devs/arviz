"""Plot posterior traces as violin plot."""
import matplotlib.pyplot as plt
import numpy as np

from ..data import convert_to_dataset
from ..stats import hpd
from .kdeplot import _fast_kde
from .plot_utils import get_bins, _scale_text, xarray_var_iter, make_label


def plot_violin(data, var_names=None, quartiles=True, credible_interval=0.94, shade=0.35,
                bw=4.5, sharey=True, figsize=None, textsize=None, ax=None, kwargs_shade=None):
    """Plot posterior of traces as violin plot.

    Notes
    -----
    If multiple chains are provided for a variable they will be combined

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names: list, optional
        List of variables to plot (defaults to None, which results in all variables plotted)
    quartiles : bool, optional
        Flag for plotting the interquartile range, in addition to the credible_interval*100%
        intervals. Defaults to True
    credible_interval : float, optional
        Credible intervals. Defaults to 0.94.
    shade : float
        Alpha blending value for the shaded area under the curve, between 0
        (no shade) and 1 (opaque). Defaults to 0
    bw : float
        Bandwidth scaling factor. Should be larger than 0. The higher this number the smoother the
        KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule of thumb
        (the default rule used by SciPy).
    figsize : tuple
        Figure size. If None, size is 5 (num of variables * 2, 5)
    textsize: int
        Text size of the point_estimates, axis ticks, and HPD. If None it will be autoscaled
        based on figsize.
    sharey : bool
        Defaults to True, violinplots share a common y-axis scale.
    ax : matplotlib axes
    kwargs_shade : dicts, optional
        Additional keywords passed to `fill_between`, or `barh` to control the shade

    Returns
    -------
    ax : matplotlib axes
    """
    data = convert_to_dataset(data, group='posterior')
    plotters = list(xarray_var_iter(data, var_names=var_names, combined=True))

    if kwargs_shade is None:
        kwargs_shade = {}

    if figsize is None:
        figsize = (len(plotters) * 2, 5)

    textsize, linewidth, _ = _scale_text(figsize, textsize=textsize)

    if ax is None:
        _, ax = plt.subplots(1, len(plotters), figsize=figsize, sharey=sharey)
    ax = np.atleast_1d(ax)

    for axind, (var_name, selection, x) in enumerate(plotters):
        val = x.flatten()
        if val[0].dtype.kind == 'i':
            cat_hist(val, shade, ax[axind], **kwargs_shade)
        else:
            _violinplot(val, shade, bw, ax[axind], **kwargs_shade)

        per = np.percentile(val, [25, 75, 50])
        hpd_intervals = hpd(val, credible_interval)

        if quartiles:
            ax[axind].plot([0, 0], per[:2], lw=linewidth*3, color='k', solid_capstyle='round')
        ax[axind].plot([0, 0], hpd_intervals, lw=linewidth, color='k', solid_capstyle='round')
        ax[axind].plot(0, per[-1], 'wo', ms=linewidth*1.5)

        ax[axind].set_xlabel(make_label(var_name, selection), fontsize=textsize)
        ax[axind].set_xticks([])
        ax[axind].tick_params(labelsize=textsize)
        ax[axind].grid(None, axis='x')

    if sharey:
        plt.subplots_adjust(wspace=0)
    else:
        plt.tight_layout()
    return ax


def _violinplot(val, shade, bw, ax, **kwargs_shade):
    """Auxiliary function to plot violinplots."""
    density, low_b, up_b = _fast_kde(val, bw=bw)
    x = np.linspace(low_b, up_b, len(density))

    x = np.concatenate([x, x[::-1]])
    density = np.concatenate([-density, density[::-1]])

    ax.fill_betweenx(x, density, alpha=shade, lw=0, **kwargs_shade)


def cat_hist(val, shade, ax, **kwargs_shade):
    """Auxiliary function to plot discrete-violinplots."""
    bins = get_bins(val)
    binned_d, _ = np.histogram(val, bins=bins, normed=True)

    bin_edges = np.linspace(np.min(val), np.max(val), len(bins))
    centers = .5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
    heights = np.diff(bin_edges)

    lefts = - .5 * binned_d
    ax.barh(centers, binned_d, height=heights, left=lefts, alpha=shade, **kwargs_shade)
