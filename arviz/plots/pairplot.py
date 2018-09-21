"""Plot a scatter or hexbin of sampled parameters."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..data import convert_to_dataset
from .kdeplot import plot_kde
from .plot_utils import _scale_text, xarray_to_ndarray, get_coords


def plot_pair(data, var_names=None, coords=None, figsize=None, textsize=None, kind='scatter',
              gridsize='auto', contour=True, fill_last=True, divergences=False, colorbar=False,
              gs=None, ax=None, divergences_kwargs=None, plot_kwargs=None):
    """
    Plot a scatter or hexbin matrix of the sampled parameters.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names
        Variables to be plotted, if None all variable are plotted
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
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
    contour : bool
        If True plot the 2D KDE using contours, otherwise plot a smooth 2D KDE. Defaults to True.
    fill_last : bool
        If True fill the last contour of the 2D KDE plot. Defaults to True.
    divergences : Boolean
        If True divergences will be plotted in a different color
    colorbar : bool
        If True a colorbar will be included as part of the plot (Defaults to False).
        Only works when kind=hexbin
    gs : Grid spec
        Matplotlib Grid spec.
    ax: axes
        Matplotlib axes
    divergences_kwargs : dicts, optional
        Additional keywords passed to ax.scatter for divergences
    plot_kwargs : dicts, optional
        Additional keywords passed to ax.scatter, az.plot_kde or ax.hexbin
    Returns
    -------
    ax : matplotlib axes
    gs : matplotlib gridspec

    """
    valid_kinds = ['scatter', 'kde', 'hexbin']
    if kind not in valid_kinds:
        raise ValueError(('Plot type {} not recognized.'
                          'Plot type must be in {}').format(kind, valid_kinds))

    if coords is None:
        coords = {}

    if plot_kwargs is None:
        plot_kwargs = {}

    # Get posterior draws and combine chains
    posterior_data = convert_to_dataset(data, group='posterior')
    _var_names, _posterior = xarray_to_ndarray(get_coords(posterior_data, coords),
                                               var_names=var_names, combined=True)

    # Get diverging draws and combine chains
    divergent_data = convert_to_dataset(data, group='sample_stats')
    _, diverging_mask = xarray_to_ndarray(divergent_data, var_names=('diverging',), combined=True)
    diverging_mask = np.squeeze(diverging_mask)

    if divergences_kwargs is None:
        divergences_kwargs = {}

    if gridsize == 'auto':
        gridsize = int(len(_posterior[0])**0.35)

    numvars = len(_var_names)

    if figsize is None:
        figsize = (2 * numvars, 2 * numvars)

    if textsize is None:
        scale_ratio = (6 / numvars) ** 0.75
        textsize, _, markersize = _scale_text(figsize, textsize=textsize, scale_ratio=scale_ratio)

    if numvars < 2:
        raise Exception('Number of variables to be plotted must be 2 or greater.')

    if numvars == 2 and ax is not None:
        if kind == 'scatter':
            ax.scatter(_posterior[0], _posterior[1], s=markersize, **plot_kwargs)
        elif kind == 'kde':
            plot_kde(_posterior[0], _posterior[1], contour=contour, fill_last=fill_last, ax=ax,
                     **plot_kwargs)
        else:
            hexbin = ax.hexbin(posterior_data[0], posterior_data[1], mincnt=1, gridsize=gridsize,
                               **plot_kwargs)
            ax.grid(False)
            if colorbar:
                cbar = plt.colorbar(hexbin, ticks=[hexbin.norm.vmin, hexbin.norm.vmax], ax=ax)
                cbar.ax.set_yticklabels(['low', 'high'], fontsize=textsize)

        if divergences:
            ax.scatter(_posterior[0][diverging_mask], _posterior[1][diverging_mask],
                       s=markersize, **divergences_kwargs)

        ax.set_xlabel('{}'.format(_var_names[0]), fontsize=textsize)
        ax.set_ylabel('{}'.format(_var_names[1]), fontsize=textsize)
        ax.tick_params(labelsize=textsize)

    if gs is None and ax is None:
        plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(numvars - 1, numvars - 1, wspace=0.05, hspace=0.05)

        axs = []
        for i in range(0, numvars - 1):
            var1 = _posterior[i]

            for j in range(i, numvars - 1):
                var2 = _posterior[j + 1]

                ax = plt.subplot(gs[j, i])

                if kind == 'scatter':
                    ax.scatter(var1, var2, s=markersize, **plot_kwargs)
                elif kind == 'kde':
                    plot_kde(var1, var2, contour=contour, fill_last=fill_last, ax=ax, **plot_kwargs)
                else:
                    ax.grid(False)
                    hexbin = ax.hexbin(var1, var2, mincnt=1, gridsize=gridsize, **plot_kwargs)
                    divider = make_axes_locatable(ax)
                    divider.append_axes('right', size='1%').set_axis_off()
                    divider.append_axes('top', size='1%').set_axis_off()

                    if i == j == 0 and colorbar:
                        cax = divider.append_axes('right', size='7%')
                        cbar = plt.colorbar(hexbin, ticks=[hexbin.norm.vmin, hexbin.norm.vmax],
                                            cax=cax)
                        cbar.ax.set_yticklabels(['low', 'high'], fontsize=textsize)

                if divergences:
                    ax.scatter(var1[diverging_mask], var2[diverging_mask],
                               s=markersize, **divergences_kwargs)

                if j + 1 != numvars - 1:
                    ax.axes.get_xaxis().set_major_formatter(NullFormatter())
                else:
                    ax.set_xlabel('{}'.format(_var_names[i]), fontsize=textsize)
                if i != 0:
                    ax.axes.get_yaxis().set_major_formatter(NullFormatter())
                else:
                    ax.set_ylabel('{}'.format(_var_names[j + 1]), fontsize=textsize)

                ax.tick_params(labelsize=textsize)
                axs.append(ax)

    plt.tight_layout()
    return ax, gs
