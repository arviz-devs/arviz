import numpy as np
import matplotlib.pyplot as plt
from .plot_utils import get_axis
from ..utils.utils import get_varnames, trace_to_dataframe


def autocorrplot(trace, varnames=None, max_lag=100, skip_first=0, symmetric_plot=False,
                 figsize=None, ax=None):
    """
    Bar plot of the autocorrelation function for a trace.

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted.
        Vector-value stochastics are handled automatically.
    max_lag : int
        Maximum lag to calculate autocorrelation. Defaults to 100.
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    symmetric_plot : boolean
        Plot from either [0, +lag] or [-lag, lag]. Defaults to False, [-, +lag].
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inches.
        Note this is not used if ax is supplied.
    ax : axes
        Matplotlib axes. Defaults to None.


    Returns
    -------
    ax : matplotlib axes
    """
    trace = trace_to_dataframe(trace, combined=False)[skip_first:]
    varnames = get_varnames(trace, varnames)

    if figsize is None:
        figsize = (12, len(varnames) * 2)

    nchains = trace.columns.value_counts()[0]
    ax = get_axis(ax, len(varnames), nchains,
                  squeeze=False, sharex=True, sharey=True, figsize=figsize)

    max_lag = min(len(trace) - 1, max_lag)

    for i, v in enumerate(varnames):
        for j in range(nchains):
            if nchains == 1:
                d = trace[v].values
            else:
                d = trace[v].values[:, j]
            ax[i, j].acorr(d, detrend=plt.mlab.detrend_mean, maxlags=max_lag)

            if j == 0:
                ax[i, j].set_ylabel("correlation")

            if i == len(varnames) - 1:
                ax[i, j].set_xlabel("lag")

            if not symmetric_plot:
                ax[i, j].set_xlim(0, max_lag)

            if nchains > 1:
                ax[i, j].set_title("{0} (chain {1})".format(v, j))
            else:
                ax[i, j].set_title(v)
    return ax
