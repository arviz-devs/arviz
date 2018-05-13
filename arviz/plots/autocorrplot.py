import matplotlib.pyplot as plt

from .plot_utils import get_axis, _scale_text
from ..utils import get_varnames, trace_to_dataframe


def autocorrplot(trace, varnames=None, max_lag=100, symmetric_plot=False, combined=False,
                 figsize=None, textsize=None, skip_first=0, ax=None):
    """
    Bar plot of the autocorrelation function for a trace.

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names, optional
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
    ax : axes, optional
        Matplotlib axes.

    Returns
    -------
    ax : matplotlib axes
    """
    trace = trace_to_dataframe(trace[skip_first:], combined=combined)
    varnames = get_varnames(trace, varnames)

    if figsize is None:
        figsize = (6, len(varnames) * 2)

    textsize, linewidth, _ = _scale_text(figsize, textsize=textsize)

    nchains = trace.columns.value_counts()[0]
    ax = get_axis(ax, len(varnames), nchains, squeeze=False, sharex=True, sharey=True,
                  figsize=figsize)

    max_lag = min(len(trace) - 1, max_lag)

    for i, v in enumerate(varnames):
        for j in range(nchains):
            if nchains == 1:
                d = trace[v].values
            else:
                d = trace[v].values[:, j]
            ax[i, j].acorr(d, detrend=plt.mlab.detrend_mean, maxlags=max_lag, lw=linewidth)

            if j == 0:
                ax[i, j].set_ylabel("correlation", fontsize=textsize)

            if i == len(varnames) - 1:
                ax[i, j].set_xlabel("lag", fontsize=textsize)

            if not symmetric_plot:
                ax[i, j].set_xlim(0, max_lag)

            if nchains > 1:
                ax[i, j].set_title("{0} (chain {1})".format(v, j), fontsize=textsize)
            else:
                ax[i, j].set_title(v, fontsize=textsize)
            ax[i, j].tick_params(labelsize=textsize)
    return ax
