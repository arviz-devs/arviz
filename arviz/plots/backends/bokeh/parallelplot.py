"""Bokeh Parallel coordinates plot."""
import bokeh.plotting as bkp
import numpy as np
from bokeh.models.tickers import FixedTicker

from ....rcparams import rcParams


def plot_parallel(ax, diverging_mask, _posterior, var_names, figsize, show):
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

    non_div = list(_posterior[:, ~diverging_mask].T)
    x_non_div = [list(range(len(non_div[0]))) for _ in range(len(non_div))]

    ax.multi_line(
        x_non_div, non_div, line_color="black", line_alpha=0.05,
    )

    if np.any(diverging_mask):
        div = list(_posterior[:, diverging_mask].T)
        x_non_div = [list(range(len(div[0]))) for _ in range(len(div))]
        ax.multi_line(x_non_div, div, color="lime", line_width=1, line_alpha=0.5)

    ax.xaxis.ticker = FixedTicker(ticks=list(range(len(var_names))))
    ax.xaxis.major_label_overrides = dict(zip(map(str, range(len(var_names))), map(str, var_names)))
    ax.xaxis.major_label_orientation = np.pi / 2

    if show:
        bkp.show(ax)

    return ax
