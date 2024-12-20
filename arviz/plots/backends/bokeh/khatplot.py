"""Bokeh pareto shape plot."""

from collections.abc import Iterable

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import Span
from matplotlib.colors import to_rgba_array

from ....stats.density_utils import histogram
from ...plot_utils import _scale_fig_size, color_from_dim, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


def plot_khat(
    hover_label,  # pylint: disable=unused-argument
    hover_format,  # pylint: disable=unused-argument
    ax,
    figsize,
    xdata,
    khats,
    good_k,
    kwargs,
    threshold,
    coord_labels,
    show_hlines,
    show_bins,
    hlines_kwargs,
    xlabels,  # pylint: disable=unused-argument
    legend,  # pylint: disable=unused-argument
    color,
    dims,
    textsize,
    markersize,  # pylint: disable=unused-argument
    n_data_points,
    bin_format,
    backend_kwargs,
    show,
):
    """Bokeh khat plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(
            ("dpi", "plot.bokeh.figure.dpi"),
        ),
        **backend_kwargs,
    }

    (figsize, *_, line_width, _) = _scale_fig_size(figsize, textsize)

    if hlines_kwargs is None:
        hlines_kwargs = {}

    if good_k is None:
        good_k = 0.7

    hlines_kwargs.setdefault("hlines", [0, good_k, 1])

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
    alphas = 0.5 + 0.2 * (khats > good_k) + 0.3 * (khats > 1)

    rgba_c = vectorized_to_hex(rgba_c)

    if ax is None:
        ax = create_axes_grid(
            1,
            figsize=figsize,
            squeeze=True,
            backend_kwargs=backend_kwargs,
        )

    if not isinstance(rgba_c, str) and isinstance(rgba_c, Iterable):
        for idx, (alpha, rgba_c_) in enumerate(zip(alphas, rgba_c)):
            ax.scatter(
                xdata[idx],
                khats[idx],
                marker="cross",
                line_color=rgba_c_,
                fill_color=rgba_c_,
                line_alpha=alpha,
                fill_alpha=alpha,
                size=10,
            )
    else:
        ax.scatter(
            xdata,
            khats,
            marker="cross",
            line_color=rgba_c,
            fill_color=rgba_c,
            size=10,
            line_alpha=alphas,
            fill_alpha=alphas,
        )

    if threshold is not None:
        idxs = xdata[khats > threshold]
        for idx in idxs:
            ax.text(x=[idx], y=[khats[idx]], text=[coord_labels[idx]])

    if show_hlines:
        for hline in hlines_kwargs.pop("hlines"):
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
        bin_edges = np.array([ymin, good_k, 1, ymax])
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
    elif ymax > 1 & threshold:
        ax.y_range._property_values["end"] = 1.1 * ymax  # pylint: disable=protected-access

    show_layout(ax, show)

    return ax
