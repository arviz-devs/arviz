import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import NullFormatter
from ..utils.utils import trace_to_dataframe, get_stats, get_varnames
from .plot_utils import _scale_text


def pairplot(trace, varnames=None, figsize=None, text_size=None, hexbin=False, gridsize='auto',
             divergences=False, skip_first=0, gs=None, ax=None, kwargs_divergences=None, **kwargs):
    """
    Plot a scatter or hexbin matrix of the sampled parameters.

    Parameters
    ----------

    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    figsize : figure size tuple
        If None, size is (8 + numvars, 8 + numvars)
    text_size: int
        Text size for labels
    hexbin : Boolean
        If True draws an hexbin plot
    gridsize : int or (int, int), optional
        Only works when hexbin is True.
        The number of hexagons in the x-direction. The corresponding number of hexagons in the
        y-direction is chosen such that the hexagons are approximately regular.
        Alternatively, gridsize can be a tuple with two elements specifying the number of hexagons
        in the x-direction and the y-direction.
    divergences : Boolean
        If True divergences will be plotted in a diferent color
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    gs : Grid spec
        Matplotlib Grid spec.
    ax: axes
        Matplotlib axes
    kwargs_divergences : dicts, optional
        Aditional keywords passed to ax.scatter for divergences
    Returns
    -------

    ax : matplotlib axes
    gs : matplotlib gridspec

    """

    if divergences:
        divergent = get_stats(trace, 'diverging')

    trace = trace_to_dataframe(trace, combined=True)[skip_first:]
    varnames = get_varnames(trace, varnames)

    if kwargs_divergences is None:
        kwargs_divergences = {}

    if gridsize == 'auto':
        gridsize = int(len(trace)*0.01)

    numvars = len(varnames)

    if figsize is None:
        figsize = (8 + numvars, 8 + numvars)

    if text_size is None:
        text_size = _scale_text(figsize, text_size=text_size)

    if numvars < 2:
        raise Exception('Number of variables to be plotted must be 2 or greater.')

    if numvars == 2 and ax is not None:
        if hexbin:
            ax.hexbin(trace[varnames[0]], trace[varnames[1]], mincnt=1, gridsize=gridsize,
                      **kwargs)
            ax.grid(False)
        else:
            ax.scatter(trace[varnames[0]], trace[varnames[1]], **kwargs)

        if divergences:
            ax.scatter(trace[varnames[0]][divergent], trace[varnames[1]][divergent],
                       **kwargs_divergences)

        ax.set_xlabel('{}'.format(varnames[0]), fontsize=text_size)
        ax.set_ylabel('{}'.format(varnames[1]), fontsize=text_size)
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
                    ax.hexbin(var1, var2, mincnt=1, gridsize=gridsize, **kwargs)
                    ax.grid(False)
                else:
                    ax.scatter(var1, var2, **kwargs)

                if divergences:
                    ax.scatter(var1[divergent], var2[divergent], **kwargs_divergences)

                if j + 1 != numvars - 1:
                    ax.axes.get_xaxis().set_major_formatter(NullFormatter())
                else:
                    ax.set_xlabel('{}'.format(varnames[i]), fontsize=text_size)
                if i != 0:
                    ax.axes.get_xaxis().set_major_formatter(NullFormatter())
                else:
                    ax.set_ylabel('{}'.format(varnames[j + 1]), fontsize=text_size)

                ax.tick_params(labelsize=text_size)

    plt.tight_layout()
    return ax, gs
