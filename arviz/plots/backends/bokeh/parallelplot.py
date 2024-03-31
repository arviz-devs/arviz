"""Bokeh Parallel coordinates plot."""

import numpy as np
from bokeh.models import DataRange1d
from bokeh.models.tickers import FixedTicker

from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


def plot_parallel(
    ax,
    colornd,  # pylint: disable=unused-argument
    colord,  # pylint: disable=unused-argument
    shadend,  # pylint: disable=unused-argument
    diverging_mask,
    posterior,
    textsize,
    var_names,
    legend,  # pylint: disable=unused-argument
    figsize,
    backend_kwargs,
    backend_config,
    show,
):
    """Bokeh parallel plot."""
    if backend_config is None:
        backend_config = {}

    backend_config = {
        **backend_kwarg_defaults(
            ("bounds_x_range", "plot.bokeh.bounds_x_range"),
            ("bounds_y_range", "plot.bokeh.bounds_y_range"),
        ),
        **backend_config,
    }

    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, *_ = _scale_fig_size(figsize, textsize, 1, 1)

    if ax is None:
        ax = create_axes_grid(
            1,
            figsize=figsize,
            squeeze=True,
            backend_kwargs=backend_kwargs,
        )

    non_div = list(posterior[:, ~diverging_mask].T)
    x_non_div = [list(range(len(non_div[0]))) for _ in range(len(non_div))]

    ax.multi_line(
        x_non_div,
        non_div,
        line_color="black",
        line_alpha=0.05,
    )

    if np.any(diverging_mask):
        div = list(posterior[:, diverging_mask].T)
        x_non_div = [list(range(len(div[0]))) for _ in range(len(div))]
        ax.multi_line(x_non_div, div, color="lime", line_width=1, line_alpha=0.5)

    ax.xaxis.ticker = FixedTicker(ticks=list(range(len(var_names))))
    ax.xaxis.major_label_overrides = dict(zip(map(str, range(len(var_names))), map(str, var_names)))
    ax.xaxis.major_label_orientation = np.pi / 2

    ax.x_range = DataRange1d(bounds=backend_config["bounds_x_range"], min_interval=2)
    ax.y_range = DataRange1d(bounds=backend_config["bounds_y_range"], min_interval=5)

    show_layout(ax, show)

    return ax
