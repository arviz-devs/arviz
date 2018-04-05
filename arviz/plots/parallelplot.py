import matplotlib.pyplot as plt
from arviz.utils import trace_to_dataframe, get_varnames, get_stats


def parallelplot(trace, varnames=None, figsize=None, textsize=14, legend=True, colornd='k',
                 colord='C1', shadend=.025, ax=None):
    """
    A parallel coordinates plot showing posterior points with and without divergences

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted. Can be used to change the order
        of the plotted variables
    figsize : figure size tuple
        If None, size is (12 x 6)
    textsize : int
        Text size of the axis ticks (Default:14)
    legend : bool
        Flag for plotting legend (defaults to True)
    colornd : valid matplotlib color
        color for non-divergent points. Defaults to 'k'
    colord : valid matplotlib color
        color for divergent points. Defaults to 'C1'
    shadend : float
        Alpha blending value for non-divergent points, between 0 (invisible) and 1 (opaque).
        Defaults to .025
        
    ax : axes
        Matplotlib axes.

    Returns
    -------
    ax : matplotlib axes
    """
    divergent = get_stats(trace, 'diverging')
    trace = trace_to_dataframe(trace)
    varnames = get_varnames(trace, varnames)

    if len(varnames) < 2:
        raise ValueError('This plot needs at least two variables')

    trace = trace[varnames]

    if figsize is None:
        figsize = (12, 6)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(trace.values[divergent == 0].T, color=colornd, alpha=shadend)
    ax.plot(trace.values[divergent == 1].T, color=colord, lw=1)

    ax.tick_params(labelsize=textsize)
    ax.set_xticks(range(trace.shape[1]))
    ax.set_xticklabels(varnames)

    if legend:
        ax.plot([], color=colornd, label='non-divergent')
        ax.plot([], color=colord, label='divergent')
        ax.legend(fontsize=textsize)

    return ax
