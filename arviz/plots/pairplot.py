"""Plot a scatter or hexbin of sampled parameters."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..data import convert_to_dataset
from .kdeplot import plot_kde
from .plot_utils import _scale_fig_size, xarray_to_ndarray, get_coords
from ..utils import _var_names


def plot_pair(
    data,
    var_names=None,
    coords=None,
    figsize=None,
    textsize=None,
    kind="scatter",
    gridsize="auto",
    contour=True,
    fill_last=True,
    divergences=False,
    colorbar=False,
    ax=None,
    divergences_kwargs=None,
    plot_kwargs=None,
):
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
    ax: axes
        Matplotlib axes
    divergences_kwargs : dicts, optional
        Additional keywords passed to ax.scatter for divergences
    plot_kwargs : dicts, optional
        Additional keywords passed to ax.plot, az.plot_kde or ax.hexbin
    Returns
    -------
    ax : matplotlib axes
    """
    var_names = _var_names(var_names)

    valid_kinds = ["scatter", "kde", "hexbin"]
    if kind not in valid_kinds:
        raise ValueError(
            ("Plot type {} not recognized." "Plot type must be in {}").format(kind, valid_kinds)
        )

    if coords is None:
        coords = {}

    if plot_kwargs is None:
        plot_kwargs = {}

    if kind == "scatter":
        plot_kwargs.setdefault("marker", ".")
        plot_kwargs.setdefault("lw", 0)

    if divergences_kwargs is None:
        divergences_kwargs = {}

    divergences_kwargs.setdefault("marker", "o")
    divergences_kwargs.setdefault("markeredgecolor", "k")
    divergences_kwargs.setdefault("color", "C1")
    divergences_kwargs.setdefault("lw", 0)

    # Get posterior draws and combine chains
    posterior_data = convert_to_dataset(data, group="posterior")
    flat_var_names, _posterior = xarray_to_ndarray(
        get_coords(posterior_data, coords), var_names=var_names, combined=True
    )

    # Get diverging draws and combine chains
    if divergences:
        divergent_data = convert_to_dataset(data, group="sample_stats")
        _, diverging_mask = xarray_to_ndarray(
            divergent_data, var_names=("diverging",), combined=True
        )
        diverging_mask = np.squeeze(diverging_mask)

    if gridsize == "auto":
        gridsize = int(len(_posterior[0]) ** 0.35)

    numvars = len(flat_var_names)

    if numvars < 2:
        raise Exception("Number of variables to be plotted must be 2 or greater.")

    elif numvars == 2:
        (figsize, ax_labelsize, _, xt_labelsize, _, _) = _scale_fig_size(
            figsize, textsize, numvars - 1, numvars - 1
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        if kind == "scatter":
            ax.plot(_posterior[0], _posterior[1], **plot_kwargs)
        elif kind == "kde":
            plot_kde(
                _posterior[0],
                _posterior[1],
                contour=contour,
                fill_last=fill_last,
                ax=ax,
                **plot_kwargs
            )
        else:
            hexbin = ax.hexbin(
                _posterior[0], _posterior[1], mincnt=1, gridsize=gridsize, **plot_kwargs
            )
            ax.grid(False)

        if kind == "hexbin" and colorbar:
            cbar = ax.figure.colorbar(hexbin, ticks=[hexbin.norm.vmin, hexbin.norm.vmax], ax=ax)
            cbar.ax.set_yticklabels(["low", "high"], fontsize=ax_labelsize)

        if divergences:
            ax.plot(
                _posterior[0][diverging_mask], _posterior[1][diverging_mask], **divergences_kwargs
            )

        ax.set_xlabel("{}".format(flat_var_names[0]), fontsize=ax_labelsize, wrap=True)
        ax.set_ylabel("{}".format(flat_var_names[1]), fontsize=ax_labelsize, wrap=True)
        ax.tick_params(labelsize=xt_labelsize)
        axs = ax

    else:
        (figsize, ax_labelsize, _, xt_labelsize, _, _) = _scale_fig_size(
            figsize, textsize, numvars - 2, numvars - 2
        )

        if ax is None:
            fig, ax = plt.subplots(
                numvars - 1, numvars - 1, figsize=figsize, constrained_layout=True
            )
        axs = []
        hexbin_values = []
        for i in range(0, numvars - 1):
            var1 = _posterior[i]

            for j in range(0, numvars - 1):
                if j < i:
                    ax[j, i].axis("off")
                    continue

                var2 = _posterior[j + 1]

                if kind == "scatter":
                    ax[j, i].plot(var1, var2, **plot_kwargs)

                elif kind == "kde":
                    plot_kde(
                        var1, var2, contour=contour, fill_last=fill_last, ax=ax[j, i], **plot_kwargs
                    )

                else:
                    ax[j, i].grid(False)
                    hexbin = ax[j, i].hexbin(var1, var2, mincnt=1, gridsize=gridsize, **plot_kwargs)
                if kind == "hexbin" and colorbar:
                    hexbin_values.append(hexbin.norm.vmin)
                    hexbin_values.append(hexbin.norm.vmax)
                    if j == i == 0 and colorbar:
                        divider = make_axes_locatable(ax[0, 1])
                        cax = divider.append_axes("left", size="7%")
                        cbar = fig.colorbar(
                            hexbin, ticks=[hexbin.norm.vmin, hexbin.norm.vmax], cax=cax
                        )
                        cbar.ax.set_yticklabels(["low", "high"], fontsize=ax_labelsize)

                if divergences:
                    ax[j, i].plot(var1[diverging_mask], var2[diverging_mask], **divergences_kwargs)

                if j + 1 != numvars - 1:
                    ax[j, i].axes.get_xaxis().set_major_formatter(NullFormatter())
                else:
                    ax[j, i].set_xlabel(
                        "{}".format(flat_var_names[i]), fontsize=ax_labelsize, wrap=True
                    )
                if i != 0:
                    ax[j, i].axes.get_yaxis().set_major_formatter(NullFormatter())
                else:
                    ax[j, i].set_ylabel(
                        "{}".format(flat_var_names[j + 1]), fontsize=ax_labelsize, wrap=True
                    )

                ax[j, i].tick_params(labelsize=xt_labelsize)
                axs.append(ax)

    return axs
