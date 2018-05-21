import matplotlib.pyplot as plt

from .plot_utils import _scale_text
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
        figsize = (12, len(varnames) * 2)

    textsize, linewidth, _ = _scale_text(figsize, textsize, 1)

    nchains = trace.columns.value_counts()[0]
    fig, ax = plt.subplots(len(varnames), nchains, squeeze=False, sharex=True, sharey=True,
                           figsize=figsize)

    max_lag = min(len(trace) - 1, max_lag)

    for i, varname in enumerate(varnames):
        for j in range(nchains):
            if nchains == 1:
                data = trace[varname].values
            else:
                data = trace[varname].values[:, j]
            ax[i, j].acorr(data, detrend=plt.mlab.detrend_mean, maxlags=max_lag, lw=linewidth)

            if not symmetric_plot:
                ax[i, j].set_xlim(0, max_lag)

            if nchains > 1:
                ax[i, j].set_title("{0} (chain {1})".format(varname, j), fontsize=textsize)
            else:
                ax[i, j].set_title(varname, fontsize=textsize)
            ax[i, j].tick_params(labelsize=textsize)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Lag", fontsize=textsize)
    plt.ylabel("Correlation", fontsize=textsize)
    return ax
