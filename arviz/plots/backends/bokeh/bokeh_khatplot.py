"""Bokeh pareto shape plot."""
from collections.abc import Iterable

import bokeh.plotting as bkp
from bokeh.models import ColumnDataSource, Span

import numpy as np

from ....rcparams import rcParams
from ....stats.stats_utils import histogram


def _plot_khat(
    ax,
    figsize,
    xdata,
    khats,
    rgba_c,
    annotate,
    coord_labels,
    show_bins,
    linewidth,
    n_data_points,
    bin_format,
    show,
):
    if ax is None:
        tools = rcParams["plot.bokeh.tools"]
        output_backend = rcParams["plot.bokeh.output_backend"]
        dpi = rcParams["plot.bokeh.figure.dpi"]
        ax = bkp.figure(
            width=int(figsize[0] * dpi),
            height=int(figsize[1] * dpi),
            output_backend=output_backend,
            tools=tools,
        )

    if not isinstance(rgba_c, str) and isinstance(rgba_c, Iterable):
        for idx, rgba_c_ in enumerate(rgba_c):
            ax.cross(xdata[idx], khats[idx], line_color=rgba_c_, fill_color=rgba_c_, size=10)
    else:
        ax.cross(xdata, khats, line_color=rgba_c, fill_color=rgba_c, size=10)

    if annotate:
        idxs = xdata[khats > 1]
        for idx in idxs:
            cds = ColumnDataSource({"x": [idx], "y": [khats[idx]], "text": [coord_labels[idx]],})
            ax.text(x="x", y="y", text="text", source=cds)

    for hline in [0, 0.5, 0.7, 1]:
        _hline = Span(
            location=hline,
            dimension="width",
            line_color="grey",
            line_width=linewidth,
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
            cds = ColumnDataSource(
                {
                    "x": [(n_data_points - 1 + xmax) / 2],
                    "y": [np.mean(bin_edges[idx : idx + 2])],
                    "text": [bin_format.format(count, count / n_data_points * 100)],
                }
            )
            ax.text(x="x", y="y", text="text", source=cds)
        ax.x_range._property_values["end"] = xmax + 1  # pylint: disable=protected-access
    ax.xaxis.axis_label = "Data Point"
    ax.yaxis.axis_label = "Shape parameter k"

    if ymin > 0:
        ax.y_range._property_values["start"] = -0.02  # pylint: disable=protected-access
    if ymax < 1:
        ax.y_range._property_values["end"] = 1.02  # pylint: disable=protected-access
    elif ymax > 1 & annotate:
        ax.y_range._property_values["end"] = 1.1 * ymax  # pylint: disable=protected-access

    if show:
        bkp.show(ax, toolbar_location="above")

    return ax
