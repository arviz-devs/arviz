"""Matplotlib kdeplot."""
from collections.abc import Iterable
import warnings

import bokeh.plotting as bkp
from bokeh.models import ColumnDataSource, Span

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ...plot_utils import set_xticklabels
from ....stats.stats_utils import histogram


def _plot_khat(
    hover_label,
    ax,
    figsize,
    xdata,
    khats,
    rgba_c,
    kwargs,
    annotate,
    coord_labels,
    ax_labelsize,
    xt_labelsize,
    show_bins,
    linewidth,
    hlines_kwargs,
    xlabels,
    legend,
    color_mapping,
    n_data_points,
    bin_format,
    show,
):
    if ax is None:
        tools = ",".join(
            [
                "pan",
                "wheel_zoom",
                "box_zoom",
                "lasso_select",
                "poly_select",
                "undo",
                "redo",
                "reset",
                "save,hover",
            ]
        )
        ax = bkp.figure(
            width=int(figsize[0] * 90),
            height=int(figsize[1] * 90),
            output_backend="webgl",
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
            ax.text(
                idx, khats[idx], coord_labels[idx],
            )

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
    xmax = len(khats) + 0.1

    if show_bins:
        bin_edges = np.array([ymin, 0.5, 0.7, 1, ymax])
        bin_edges = bin_edges[(bin_edges >= ymin) & (bin_edges <= ymax)]
        _, hist, _ = histogram(khats, bin_edges)
        for idx, count in enumerate(hist):
            cds = ColumnDataSource(
                {
                    "x": [(n_data_points - 1 + xmax) / 2],
                    "y": [np.mean(bin_edges[idx : idx + 2])],
                    "text": [bin_format.format(count, count / n_data_points * 100)],
                }
            )
            ax.text(x="x", y="y", text="text", source=cds)
    ax.xaxis.axis_label = "Data Point"
    ax.yaxis.axis_label = "Shape parameter k"

    if show:
        bkp.show(ax)

    return ax
