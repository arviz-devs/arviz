"""Bokeh dotplot."""

import math
import warnings
import numpy as np

from ...plot_utils import _scale_fig_size, vectorized_to_hex
from .. import show_layout
from . import create_axes_grid
from ...plot_utils import plot_point_interval
from ...dotplot import wilkinson_algorithm, layout_stacks


def plot_dot(
    values,
    binwidth,
    dotsize,
    stackratio,
    hdi_prob,
    quartiles,
    rotated,
    dotcolor,
    intervalcolor,
    markersize,
    markercolor,
    marker,
    figsize,
    linewidth,
    point_estimate,
    nquantiles,
    point_interval,
    ax,
    show,
    backend_kwargs,
    plot_kwargs,
):
    """Bokeh dotplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs.setdefault("match_aspect", True)

    (figsize, _, _, _, auto_linewidth, auto_markersize) = _scale_fig_size(figsize, None)
    dotcolor = vectorized_to_hex(dotcolor)
    intervalcolor = vectorized_to_hex(intervalcolor)
    markercolor = vectorized_to_hex(markercolor)

    if plot_kwargs is None:
        plot_kwargs = {}
    else:
        plot_kwargs = plot_kwargs.copy()
    plot_kwargs.setdefault("color", dotcolor)
    plot_kwargs.setdefault("marker", "circle")

    if linewidth is None:
        linewidth = auto_linewidth

    if markersize is None:
        markersize = auto_markersize

    if ax is None:
        ax = create_axes_grid(
            1,
            figsize=figsize,
            squeeze=True,
            backend_kwargs=backend_kwargs,
        )

    if point_interval:
        ax = plot_point_interval(
            ax,
            values,
            point_estimate,
            hdi_prob,
            quartiles,
            linewidth,
            markersize,
            markercolor,
            marker,
            rotated,
            intervalcolor,
            "bokeh",
        )

    if nquantiles > values.shape[0]:
        warnings.warn(
            "nquantiles must be less than or equal to the number of data points", UserWarning
        )
        nquantiles = values.shape[0]
    else:
        qlist = np.linspace(1 / (2 * nquantiles), 1 - 1 / (2 * nquantiles), nquantiles)
        values = np.quantile(values, qlist)

    if binwidth is None:
        binwidth = math.sqrt((values[-1] - values[0] + 1) ** 2 / (2 * nquantiles * np.pi))

    ## Wilkinson's Algorithm
    stack_locs, stack_count = wilkinson_algorithm(values, binwidth)
    x, y = layout_stacks(stack_locs, stack_count, binwidth, stackratio, rotated)

    ax.scatter(x, y, radius=dotsize * (binwidth / 2), **plot_kwargs, radius_dimension="y")
    if rotated:
        ax.xaxis.major_tick_line_color = None
        ax.xaxis.minor_tick_line_color = None
        ax.xaxis.major_label_text_font_size = "0pt"
    else:
        ax.yaxis.major_tick_line_color = None
        ax.yaxis.minor_tick_line_color = None
        ax.yaxis.major_label_text_font_size = "0pt"

    show_layout(ax, show)

    return ax
