"""Pareto tail indices plot."""
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array
import matplotlib.cm as cm
import numpy as np
from xarray import DataArray

from .plot_utils import (
    _scale_fig_size,
    get_coords,
    color_from_dim,
    format_coords_as_labels,
    get_plotting_function,
)
from ..stats import ELPDData


def plot_khat(
    khats,
    color=None,
    xlabels=False,
    show_bins=False,
    bin_format="{1:.1f}%",
    annotate=False,
    hover_label=False,
    hover_format="{1}",
    figsize=None,
    textsize=None,
    coords=None,
    legend=False,
    markersize=None,
    ax=None,
    hlines_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
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
    show_bins : bool, optional
        Show the number of khats which fall in each bin.
    bin_format : str, optional
        The string is used as formatting guide calling ``bin_format.format(count, pct)``.
    annotate : bool, optional
        Show the labels of k values larger than 1.
    hover_label : bool, optional
        Show the datapoint label when hovering over it with the mouse. Requires an interactive
        backend.
    hover_format : str, optional
        String used to format the hover label via ``hover_format.format(idx, coord_label)``
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
        Matplotlib axes or bokeh figures.
    hlines_kwargs: dictionary, optional
        Additional keywords passed to ax.hlines.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show : bool, optional
        Call backend show function.
    kwargs :
        Additional keywords passed to ax.scatter.

    Returns
    -------
    axes : matplotlib axes or bokeh figures

    Examples
    --------
    Plot estimated pareto shape parameters showing how many fall in each category.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> radon = az.load_arviz_data("radon")
        >>> loo_radon = az.loo(radon, pointwise=True)
        >>> az.plot_khat(loo_radon, show_bins=True)

    Show xlabels

    .. plot::
        :context: close-figs

        >>> centered_eight = az.load_arviz_data("centered_eight")
        >>> khats = az.loo(centered_eight, pointwise=True).pareto_k
        >>> az.plot_khat(khats, xlabels=True, annotate=True)

    Use coord values to create color mapping

    .. plot::
        :context: close-figs

        >>> az.plot_khat(loo_radon, color="observed_county", cmap="tab20")

    Use custom color scheme

    .. plot::
        :context: close-figs

        >>> counties = radon.posterior.observed_county.values
        >>> colors = [
        ...     "blue" if county[-1] in ("A", "N") else "green" for county in counties
        ... ]
        >>> az.plot_khat(loo_radon, color=colors)

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

    n_data_points = khats.size
    xdata = np.arange(n_data_points)
    if isinstance(khats, DataArray):
        coord_labels = format_coords_as_labels(khats)
    else:
        coord_labels = xdata.astype(str)

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, scaled_markersize) = _scale_fig_size(
        figsize, textsize
    )

    if markersize is None:
        markersize = scaled_markersize ** 2  # s in scatter plot mus be markersize square
        # for dots to have the same size
    kwargs.setdefault("s", markersize)
    kwargs.setdefault("marker", "+")
    color_mapping = None
    cmap = None
    if isinstance(color, str):
        if color in dims:
            colors, color_mapping = color_from_dim(khats, color)
            cmap_name = kwargs.get("cmap", plt.rcParams["image.cmap"])
            cmap = getattr(cm, cmap_name)
            rgba_c = cmap(colors)
        else:
            legend = False
            rgba_c = to_rgba_array(np.full(n_data_points, color))
    else:
        legend = False
        try:
            rgba_c = to_rgba_array(color)
        except ValueError:
            cmap_name = kwargs.get("cmap", plt.rcParams["image.cmap"])
            cmap = getattr(cm, cmap_name)
            rgba_c = cmap(color)

    khats = khats if isinstance(khats, np.ndarray) else khats.values.flatten()
    alphas = 0.5 + 0.2 * (khats > 0.5) + 0.3 * (khats > 1)
    rgba_c[:, 3] = alphas

    plot_khat_kwargs = dict(
        hover_label=hover_label,
        hover_format=hover_format,
        ax=ax,
        figsize=figsize,
        xdata=xdata,
        khats=khats,
        rgba_c=rgba_c,
        kwargs=kwargs,
        annotate=annotate,
        coord_labels=coord_labels,
        ax_labelsize=ax_labelsize,
        xt_labelsize=xt_labelsize,
        show_bins=show_bins,
        linewidth=linewidth,
        hlines_kwargs=hlines_kwargs,
        xlabels=xlabels,
        legend=legend,
        color_mapping=color_mapping,
        cmap=cmap,
        color=color,
        n_data_points=n_data_points,
        bin_format=bin_format,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend == "bokeh":

        plot_khat_kwargs.pop("hover_label")
        plot_khat_kwargs.pop("hover_format")
        plot_khat_kwargs.pop("kwargs")
        plot_khat_kwargs.pop("ax_labelsize")
        plot_khat_kwargs.pop("xt_labelsize")
        plot_khat_kwargs.pop("hlines_kwargs")
        plot_khat_kwargs.pop("xlabels")
        plot_khat_kwargs.pop("legend")
        plot_khat_kwargs.pop("color_mapping")
        plot_khat_kwargs.pop("cmap")
        plot_khat_kwargs.pop("color")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_khat", "khatplot", backend)
    axes = plot(**plot_khat_kwargs)
    return axes
