"""Pareto tail indices plot."""
import warnings
import matplotlib as mpl
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
    set_xticklabels,
)
from ..stats import ELPDData
from ..utils import conditional_jit


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
    if hover_label and mpl.get_backend() not in mpl.rcsetup.interactive_bk:
        hover_label = False
        warnings.warn(
            "hover labels are only available with interactive backends. To switch to an "
            "interactive backend from ipython or jupyter, use `%matplotlib` there should be "
            "no need to restart the kernel. For other cases, see "
            "https://matplotlib.org/3.1.0/tutorials/introductory/usage.html#backends",
            UserWarning,
        )

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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=not xlabels)
    else:
        fig = ax.get_figure()

    khats = khats if isinstance(khats, np.ndarray) else khats.values.flatten()
    alphas = 0.5 + 0.2 * (khats > 0.5) + 0.3 * (khats > 1)
    rgba_c[:, 3] = alphas
    sc_plot = ax.scatter(xdata, khats, c=rgba_c, **kwargs)
    if annotate:
        idxs = xdata[khats > 1]
        for idx in idxs:
            ax.text(
                idx,
                khats[idx],
                coord_labels[idx],
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=0.8 * xt_labelsize,
            )

    xmin, xmax = ax.get_xlim()
    if show_bins:
        xmax += n_data_points / 12
    ylims1 = ax.get_ylim()
    ax.hlines([0, 0.5, 0.7, 1], xmin=xmin, xmax=xmax, linewidth=linewidth, **hlines_kwargs)
    ylims2 = ax.get_ylim()
    ymin = min(ylims1[0], ylims2[0])
    ymax = min(ylims1[1], ylims2[1])
    if show_bins:
        bin_edges = np.array([ymin, 0.5, 0.7, 1, ymax])
        bin_edges = bin_edges[(bin_edges >= ymin) & (bin_edges <= ymax)]
        hist, _ = _khat_histogram(khats, bin_edges)
        for idx, count in enumerate(hist):
            ax.text(
                (n_data_points - 1 + xmax) / 2,
                np.mean(bin_edges[idx : idx + 2]),
                bin_format.format(count, count / n_data_points * 100),
                horizontalalignment="center",
                verticalalignment="center",
            )
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel("Data Point", fontsize=ax_labelsize)
    ax.set_ylabel(r"Shape parameter k", fontsize=ax_labelsize)
    ax.tick_params(labelsize=xt_labelsize)
    if xlabels:
        set_xticklabels(ax, coord_labels)
        fig.autofmt_xdate()
        fig.tight_layout()
    if legend:
        ncols = len(color_mapping) // 6 + 1
        for label, float_color in color_mapping.items():
            ax.scatter([], [], c=[cmap(float_color)], label=label, **kwargs)
        ax.legend(ncol=ncols, title=color)

    if hover_label and mpl.get_backend() in mpl.rcsetup.interactive_bk:
        _make_hover_annotation(fig, ax, sc_plot, coord_labels, rgba_c, hover_format)

    return ax


def _make_hover_annotation(fig, ax, sc_plot, coord_labels, rgba_c, hover_format):
    """Show data point label when hovering over it with mouse."""
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(0, 0),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", alpha=0.4),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)
    xmid = np.mean(ax.get_xlim())
    ymid = np.mean(ax.get_ylim())
    offset = 10

    def update_annot(ind):

        idx = ind["ind"][0]
        pos = sc_plot.get_offsets()[idx]
        annot_text = hover_format.format(idx, coord_labels[idx])
        annot.xy = pos
        annot.set_position(
            (-offset if pos[0] > xmid else offset, -offset if pos[1] > ymid else offset)
        )
        annot.set_text(annot_text)
        annot.get_bbox_patch().set_facecolor(rgba_c[idx])
        annot.set_ha("right" if pos[0] > xmid else "left")
        annot.set_va("top" if pos[1] > ymid else "bottom")

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc_plot.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


@conditional_jit
def _khat_histogram(data, bin_edges):
    return np.histogram(data, bins=bin_edges)
