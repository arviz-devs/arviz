import numpy as np
import matplotlib.pyplot as plt

from .kdeplot import fast_kde
from .plot_utils import get_bins, _scale_text
from ..stats import hpd
from ..utils import get_varnames, trace_to_dataframe


def violintraceplot(trace, varnames=None, quartiles=True, alpha=0.05, shade=0.35, bw=4.5,
                    sharey=True, figsize=None, textsize=None, skip_first=0, ax=None,
                    kwargs_shade=None):
    """
    Violinplot

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames: list, optional
        List of variables to plot (defaults to None, which results in all variables plotted)
    quartiles : bool, optional
        Flag for plotting the interquartile range, in addition to the (1-alpha)*100% intervals.
        Defaults to True
    alpha : float, optional
        Alpha value for (1-alpha)*100% credible intervals. Defaults to 0.05.
    shade : float
        Alpha blending value for the shaded area under the curve, between 0
        (no shade) and 1 (opaque). Defaults to 0
    bw : float
        Bandwidth scaling factor. Should be larger than 0. The higher this number the smoother the
        KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule of thumb
        (the default rule used by SciPy).
    sharey : bool
        Defaults to True, violinplots share a common y-axis scale.
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    ax : matplotlib axes
    kwargs_shade : dicts, optional
        Additional keywords passed to `fill_between`, or `barh` to control the shade
    Returns
    ----------
    ax : matplotlib axes

    """
    trace = trace_to_dataframe(trace[skip_first:], combined=True)
    varnames = get_varnames(trace, varnames)
    trace = trace[varnames]

    if kwargs_shade is None:
        kwargs_shade = {}

    if figsize is None:
        figsize = (len(varnames) * 2, 5)

    textsize, linewidth, _ = _scale_text(figsize, textsize=textsize)

    if ax is None:
        _, ax = plt.subplots(1, len(varnames), figsize=figsize, sharey=sharey)
    ax = np.atleast_1d(ax)

    names = trace.columns.values

    for axind, var in enumerate(trace.columns):
        val = trace[var]
        if val[0].dtype.kind == 'i':
            cat_hist(val, shade, ax[axind], **kwargs_shade)
        else:
            _violinplot(val, shade, bw, ax[axind], **kwargs_shade)

        per = np.percentile(val, [25, 75, 50])
        hpd_intervals = hpd(val, alpha)

        if quartiles:
            ax[axind].plot([0, 0], per[:2], lw=linewidth*3, color='k', solid_capstyle='round')
        ax[axind].plot([0, 0], hpd_intervals, lw=linewidth, color='k', solid_capstyle='round')
        ax[axind].plot(0, per[-1], 'wo', ms=linewidth*1.5)

        ax[axind].set_xlabel(names[axind], fontsize=textsize)
        ax[axind].set_xticks([])
        ax[axind].tick_params(labelsize=textsize)
        ax[axind].grid(None, axis='x')
        #ax[axind].set_xlim(-np.max(density)*5, np.max(density)*5)

    if sharey:
        plt.subplots_adjust(wspace=0)
    else:
        plt.tight_layout()
    return ax


def _violinplot(val, shade, bw, ax, **kwargs_shade):
    """
    Auxiliar function to plot violinplots
    """
    density, low_b, up_b = fast_kde(val, bw=bw)
    x = np.linspace(low_b, up_b, len(density))

    x = np.concatenate([x, x[::-1]])
    density = np.concatenate([-density, density[::-1]])

    ax.fill_betweenx(x, density, alpha=shade, lw=0, **kwargs_shade)



def cat_hist(val, shade, ax, **kwargs_shade):
    """
    Auxiliar function to plot discrete-violinplots
    """
    bins = get_bins(val)
    binned_d, _ = np.histogram(val, bins=bins, normed=True)

    bin_edges = np.linspace(np.min(val), np.max(val), len(bins))
    centers = .5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
    heights = np.diff(bin_edges)

    lefts = - .5 * binned_d
    ax.barh(centers, binned_d, height=heights, left=lefts, alpha=shade, **kwargs_shade)
