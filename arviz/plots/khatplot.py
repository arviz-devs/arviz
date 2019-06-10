"""Pareto tail indices plot."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba_array
import matplotlib.cm as cm
import numpy as np
from xarray import DataArray

from .plot_utils import (
    _scale_fig_size,
    get_coords,
    color_from_dim,
    format_coords_as_labels,
    set_xticklabels,
)
from ..stats import ELPDData


def plot_khat(
    khats,
    color=None,
    xlabels=False,
    figsize=None,
    textsize=None,
    coords=None,
    legend=False,
    markersize=None,
    ax=None,
    hlines_kwargs=None,
    **kwargs
):
    """
    Plot Pareto tail indices.

    Parameters
    ----------
    khats : ELPDData cointaining pareto shapes information or array
        Pareto tail indices.
    color : str or array_like, optional
        Colors of the scatter plot, if color is a str all dots will have the same color,
        if it is the size of the observations, each dot will have the specified color,
        otherwise, it will be interpreted as a list of the dims to be used for the color code
    xlabels : bool, optional
        Use coords as xticklabels
    figsize : tuple, optional
        Figure size. If None it will be defined automatically.
    textsize: float, optional
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    coords : mapping, optional
        Coordinates of points to plot. **All** values are used for computation, but only a
        a subset can be plotted for convenience.
    legend : bool, optional
        Include a legend to the plot. Only taken into account when color argument is a dim name.
    markersize: int, optional
        markersize for scatter plot. Defaults to `None` in which case it will
        be chosen based on autoscaling for figsize.
    ax: axes, optional
      Matplotlib axes
    hlines_kwargs: dictionary, optional
      Additional keywords passed to ax.hlines
    kwargs :
      Additional keywords passed to ax.scatter

    Returns
    -------
    ax : axes
      Matplotlib axes.

    Examples
    --------
    Plot a default khat plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> centered_eight = az.load_arviz_data('centered_eight')
        >>> pareto_k = az.loo(centered_eight, pointwise=True)['pareto_k']
        >>> az.plot_khat(pareto_k)

    """
    if hlines_kwargs is None:
        hlines_kwargs = {}
    hlines_kwargs.setdefault("linestyle", [":", "-.", "--", "-"])
    hlines_kwargs.setdefault("alpha", 0.7)
    hlines_kwargs.setdefault("zorder", -1)
    hlines_kwargs.setdefault("color", "C1")

    if coords is None:
        coords = {}

    if color is None:
        color = "C0"

    if isinstance(khats, np.ndarray):
        khats = khats.flatten()
        xlabels = False
        legend = False
        dims = []
    else:
        if isinstance(khats, ELPDData):
            khats = khats.pareto_k
        if not isinstance(khats, DataArray):
            raise ValueError("Incorrect khat data input. Check the documentation")

        khats = get_coords(khats, coords)
        dims = khats.dims

    if xlabels:
        coord_labels = format_coords_as_labels(khats)

    n_data_points = khats.size

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, scaled_markersize) = _scale_fig_size(
        figsize, textsize
    )

    if markersize is None:
        markersize = scaled_markersize ** 2  # s in scatter plot mus be markersize square
        # for dots to have the same size
    kwargs.setdefault("s", markersize)
    kwargs.setdefault("marker", "+")

    if isinstance(color, str):
        if color in dims:
            colors, color_mapping = color_from_dim(khats, color)
            cmap_name = kwargs.pop("cmap", plt.rcParams["image.cmap"])
            cmap = getattr(cm, cmap_name)
            if legend:
                msize = np.sqrt(kwargs.pop("s", markersize))
                handles = [
                    Line2D([], [], color=cmap(float_color), label=coord, ms=msize, lw=0, **kwargs)
                    for coord, float_color in color_mapping.items()
                ]
                kwargs.setdefault("cmap", cmap_name)
                kwargs.setdefault("s", msize ** 2)
            rgba_c = cmap(colors)
        else:
            legend = False
            rgba_c = to_rgba_array(np.full(n_data_points, color))
    else:
        legend = False
        if len(color.shape) == 1 and len(color) == n_data_points:
            cmap_name = kwargs.get("cmap", plt.rcParams["image.cmap"])
            cmap = getattr(cm, cmap_name)
            rgba_c = cmap(color)
        else:
            rgba_c = to_rgba_array(color)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=(not xlabels and not legend))

    khats = khats if isinstance(khats, np.ndarray) else khats.values.flatten()
    alphas = 0.5 + 0.2 * (khats > 0.5) + 0.3 * (khats > 1)
    rgba_c[:, 3] = alphas
    ax.scatter(np.arange(n_data_points), khats, c=rgba_c, **kwargs)

    xlims = ax.get_xlim()
    ylims1 = ax.get_ylim()
    ax.hlines([0, 0.5, 0.7, 1], xmin=xlims[0], xmax=xlims[1], linewidth=linewidth, **hlines_kwargs)
    ylims2 = ax.get_ylim()
    ax.set_xlim(xlims)
    ax.set_ylim(min(ylims1[0], ylims2[0]), min(ylims1[1], ylims2[1]))

    ax.set_xlabel("Data Point", fontsize=ax_labelsize)
    ax.set_ylabel(r"Shape parameter k", fontsize=ax_labelsize)
    ax.tick_params(labelsize=xt_labelsize)
    if xlabels:
        set_xticklabels(ax, coord_labels)
        fig.autofmt_xdate()
    if legend:
        ncols = len(handles) // 6 + 1
        ax.legend(handles=handles, ncol=ncols, title=color)
    if xlabels or legend:
        fig.tight_layout()
    return ax
