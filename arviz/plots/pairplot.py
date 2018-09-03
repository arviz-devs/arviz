"""Plot a scatter or hexbin of sampled parameters."""
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..utils import trace_to_dataframe, get_stats, get_varnames
from .plot_utils import _scale_text


def pairplot(trace, varnames=None, figsize=None, textsize=None, kind='scatter', gridsize='auto',
             colorbar=False, divergences=False, skip_first=0, gs=None, ax=None,
             kwargs_divergences=None, **kwargs):
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
    textsize: int
        Text size for labels. If None it will be autoscaled based on figsize.
    kind : str
        Type of plot to display (kde or hexbin)
    gridsize : int or (int, int), optional
        Only works for kind=hexbin.
        The number of hexagons in the x-direction. The corresponding number of hexagons in the
        y-direction is chosen such that the hexagons are approximately regular.
        Alternatively, gridsize can be a tuple with two elements specifying the number of hexagons
        in the x-direction and the y-direction.
    colorbar : bool
        If True a colorbar will be included as part of the plot (Defaults to False).
        Only works when kind=hexbin
    divergences : Boolean
        If True divergences will be plotted in a diferent color
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    gs : Grid spec
        Matplotlib Grid spec.
    kwargs_divergences : dicts, optional
        Aditional keywords passed to ax.scatter for divergences
    ax: axes
        Matplotlib axes

    Returns
    -------
    ax : matplotlib axes
    gs : matplotlib gridspec

    """
    if kind not in ['scatter', 'hexbin']:
        raise ValueError('Plot type {} not recognized.'.format(kind))

    if divergences:
        divergent = get_stats(trace[skip_first:], 'diverging')

    trace = trace_to_dataframe(trace[skip_first:], combined=True)
    varnames = get_varnames(trace, varnames)

    if kwargs_divergences is None:
        kwargs_divergences = {}

    if gridsize == 'auto':
        gridsize = int(len(trace)**0.35)

    numvars = len(varnames)

    if figsize is None:
        figsize = (2 * numvars, 2 * numvars)

    if textsize is None:
        scale_ratio = (6 / numvars) ** 0.75
        textsize, _, markersize = _scale_text(figsize, textsize=textsize, scale_ratio=scale_ratio)

    if numvars < 2:
        raise Exception('Number of variables to be plotted must be 2 or greater.')

    if numvars == 2 and ax is not None:
        if kind == 'scatter':
            ax.scatter(trace[varnames[0]], trace[varnames[1]], s=markersize, **kwargs)
        else:
            hexbin = ax.hexbin(trace[varnames[0]], trace[varnames[1]], mincnt=1, gridsize=gridsize,
                               **kwargs)
            ax.grid(False)
            if colorbar:
                cbar = plt.colorbar(hexbin, ticks=[hexbin.norm.vmin, hexbin.norm.vmax], ax=ax)
                cbar.ax.set_yticklabels(['low', 'high'], fontsize=textsize)

        if divergences:
            ax.scatter(trace[varnames[0]][divergent], trace[varnames[1]][divergent],
                       s=markersize, **kwargs_divergences)

        ax.set_xlabel('{}'.format(varnames[0]), fontsize=textsize)
        ax.set_ylabel('{}'.format(varnames[1]), fontsize=textsize)
        ax.tick_params(labelsize=textsize)

    if gs is None and ax is None:
        plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(numvars - 1, numvars - 1)

        axs = []
        for i in range(0, numvars - 1):
            var1 = trace[varnames[i]]

            for j in range(i, numvars - 1):
                var2 = trace[varnames[j + 1]]

                ax = plt.subplot(gs[j, i])

                if kind == 'scatter':
                    ax.scatter(var1, var2, s=markersize, **kwargs)
                else:
                    ax.grid(False)
                    if i == j == 0 and colorbar:
                        hexbin = ax.hexbin(var1, var2, mincnt=1, gridsize=gridsize, **kwargs)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes('right', size='7%', pad=0.1)
                        cbar = plt.colorbar(hexbin,
                                            ticks=[hexbin.norm.vmin, hexbin.norm.vmax],
                                            cax=cax)
                        cbar.ax.set_yticklabels(['low', 'high'], fontsize=textsize)
                        divider.append_axes('top', size='7%', pad=0.1).set_axis_off()

                    else:
                        ax.hexbin(var1, var2, mincnt=1, gridsize=gridsize, **kwargs)
                        divider = make_axes_locatable(ax)
                        divider.append_axes('right', size='7%', pad=0.1).set_axis_off()
                        divider.append_axes('top', size='7%', pad=0.1).set_axis_off()

                if divergences:
                    ax.scatter(var1[divergent], var2[divergent], s=markersize, **kwargs_divergences)

                if j + 1 != numvars - 1:
                    ax.axes.get_xaxis().set_major_formatter(NullFormatter())
                else:
                    ax.set_xlabel('{}'.format(varnames[i]), fontsize=textsize)
                if i != 0:
                    ax.axes.get_yaxis().set_major_formatter(NullFormatter())
                else:
                    ax.set_ylabel('{}'.format(varnames[j + 1]), fontsize=textsize)

                ax.tick_params(labelsize=textsize)
                axs.append(ax)

    plt.tight_layout()
    return ax, gs
