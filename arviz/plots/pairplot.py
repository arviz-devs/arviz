import warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
from ..utils.utils import trace_to_dataframe, get_stats
from .plot_utils import _scale_text


def pairplot(trace, varnames=None, figsize=None, text_size=None,
             gs=None, ax=None, hexbin=False, divergences=False, kwargs_divergences=None,
             skip_first=0, **kwargs):
    """
    Plot a scatter or hexbin matrix of the sampled parameters.

    Parameters
    ----------

    trace : result of MCMC run
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    figsize : figure size tuple
        If None, size is (8 + numvars, 8 + numvars)
    text_size: int
        Text size for labels
    gs : Grid spec
        Matplotlib Grid spec.
    ax: axes
        Matplotlib axes
    hexbin : Boolean
        If True draws an hexbin plot
    divergences : Boolean
        If True divergences will be plotted in a diferent color
    kwargs_divergences : dicts, optional
        Aditional keywords passed to ax.scatter for divergences
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    Returns
    -------

    ax : matplotlib axes
    gs : matplotlib gridspec

    """

    if divergences:
        divergent = get_stats(trace, 'diverging')

    trace = trace_to_dataframe(trace, combined=True)[skip_first:]

    if varnames is None:
        varnames = trace.columns

    if text_size is None:
        text_size = _scale_text(figsize, text_size=text_size)

    if kwargs_divergences is None:
        kwargs_divergences = {}

    numvars = len(varnames)

    if figsize is None:
        figsize = (8 + numvars, 8 + numvars)

    if numvars < 2:
        raise Exception(
            'Number of variables to be plotted must be 2 or greater.')

    if numvars == 2 and ax is not None:
        if hexbin:
            ax.hexbin(trace[varnames[0]],
                      trace[varnames[1]], mincnt=1, **kwargs)
        else:
            ax.scatter(trace[varnames[0]],
                       trace[varnames[1]], **kwargs)

        if divergences:
            ax.scatter(trace[varnames[0]][divergent],
                       trace[varnames[1]][divergent], **kwargs_divergences)

        ax.set_xlabel('{}'.format(varnames[0]),
                      fontsize=text_size)
        ax.set_ylabel('{}'.format(
            varnames[1]), fontsize=text_size)
        ax.tick_params(labelsize=text_size)

    if gs is None and ax is None:
        plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(numvars - 1, numvars - 1)

        for i in range(0, numvars - 1):
            var1 = trace[varnames[i]]

            for j in range(i, numvars - 1):
                var2 = trace[varnames[j + 1]]

                ax = plt.subplot(gs[j, i])

                if hexbin:
                    ax.hexbin(var1, var2, mincnt=1, **kwargs)
                else:
                    ax.scatter(var1, var2, **kwargs)

                if divergences:
                    ax.scatter(var1[divergent],
                               var2[divergent],
                               **kwargs_divergences)

                if j + 1 != numvars - 1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel('{}'.format(varnames[i]),
                                  fontsize=text_size)
                if i != 0:
                    ax.set_yticks([])
                else:
                    ax.set_ylabel('{}'.format(
                        varnames[j + 1]), fontsize=text_size)

                ax.tick_params(labelsize=text_size)

    plt.tight_layout()
    return ax, gs
