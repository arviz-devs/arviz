"""Bokeh pareto shape plot."""
from collections.abc import Iterable

import bokeh.plotting as bkp
from bokeh.models import Span
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array
import matplotlib.cm as cm
import numpy as np

from . import backend_kwarg_defaults
from .. import show_layout
from ...plot_utils import (
    _scale_fig_size,
    vectorized_to_hex,
    color_from_dim,
)
from ....stats.stats_utils import histogram


def plot_khat(
    hover_label,
    hover_format,
    ax,
    figsize,
    xdata,
    khats,
    kwargs,
    annotate,
    coord_labels,
    show_bins,
    hlines_kwargs,
    xlabels,
    legend,
    color,
    dims,
    textsize,
    markersize,
    n_data_points,
    bin_format,
    backend_kwargs,
    show,
):
    """Bokeh khat plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(("dpi", "plot.bokeh.figure.dpi"),),
        **backend_kwargs,
    }
    dpi = backend_kwargs.pop("dpi")

    (figsize, *_, line_width, _) = _scale_fig_size(figsize, textsize)

    cmap = None
    if isinstance(color, str):
        if color in dims:
            colors, _ = color_from_dim(khats, color)
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
    rgba_c = vectorized_to_hex(rgba_c)

    if ax is None:
        backend_kwargs.setdefault("width", int(figsize[0] * dpi))
        backend_kwargs.setdefault("height", int(figsize[1] * dpi))
        ax = bkp.figure(**backend_kwargs)

    if not isinstance(rgba_c, str) and isinstance(rgba_c, Iterable):
        for idx, rgba_c_ in enumerate(rgba_c):
            ax.cross(xdata[idx], khats[idx], line_color=rgba_c_, fill_color=rgba_c_, size=10)
    else:
        ax.cross(xdata, khats, line_color=rgba_c, fill_color=rgba_c, size=10)

    if annotate:
        idxs = xdata[khats > 1]
        for idx in idxs:
            ax.text(x=[idx], y=[khats[idx]], text=[coord_labels[idx]])

    for hline in [0, 0.5, 0.7, 1]:
        _hline = Span(
            location=hline,
            dimension="width",
            line_color="grey",
            line_width=line_width,
            line_dash="dashed",
        )

        ax.renderers.append(_hline)

    ymin = min(khats)
    ymax = max(khats)
    xmax = len(khats)

    if show_bins:
        bin_edges = np.array([ymin, 0.5, 0.7, 1, ymax])
        bin_edges = bin_edges[(bin_edges >= ymin) & (bin_edges <= ymax)]
        hist, _, _ = histogram(khats, bin_edges)
        for idx, count in enumerate(hist):
            ax.text(
                x=[(n_data_points - 1 + xmax) / 2],
                y=[np.mean(bin_edges[idx : idx + 2])],
                text=[bin_format.format(count, count / n_data_points * 100)],
            )
        ax.x_range._property_values["end"] = xmax + 1  # pylint: disable=protected-access
    ax.xaxis.axis_label = "Data Point"
    ax.yaxis.axis_label = "Shape parameter k"

    if ymin > 0:
        ax.y_range._property_values["start"] = -0.02  # pylint: disable=protected-access
    if ymax < 1:
        ax.y_range._property_values["end"] = 1.02  # pylint: disable=protected-access
    elif ymax > 1 & annotate:
        ax.y_range._property_values["end"] = 1.1 * ymax  # pylint: disable=protected-access

    show_layout(ax, show)

    return ax
