"""Bokeh hpdplot."""
from itertools import cycle

import bokeh.plotting as bkp
from matplotlib.pyplot import rcParams as mpl_rcParams
import numpy as np

from ....rcparams import rcParams


def _plot_hpdplot(ax, x_data, y_data, plot_kwargs, fill_kwargs, show):
    if ax is None:
        tools = rcParams["plot.bokeh.tools"]
        output_backend = rcParams["plot.bokeh.output_backend"]
        ax = bkp.figure(
            width=rcParams["plot.bokeh.figure.width"],
            height=rcParams["plot.bokeh.figure.height"],
            output_backend=output_backend,
            tools=tools,
        )

    color = plot_kwargs.pop("color")
    if len(color) == 2 and color[0] == "C":
        color = [
            prop
            for _, prop in zip(
                range(int(color[1:])), cycle(mpl_rcParams["axes.prop_cycle"].by_key()["color"])
            )
        ][-1]
    plot_kwargs.setdefault("line_color", color)
    plot_kwargs.setdefault("line_alpha", plot_kwargs.pop("alpha", 0))

    color = fill_kwargs.pop("color")
    if len(color) == 2 and color[0] == "C":
        color = [
            prop
            for _, prop in zip(
                range(int(color[1:])), cycle(mpl_rcParams["axes.prop_cycle"].by_key()["color"])
            )
        ][-1]
    fill_kwargs.setdefault("fill_color", color)
    fill_kwargs.setdefault("fill_alpha", fill_kwargs.pop("alpha", 0))

    ax.patch(
        np.concatenate((x_data, x_data[::-1])),
        np.concatenate((y_data[:, 0], y_data[:, 1][::-1])),
        **fill_kwargs,
        **plot_kwargs
    )

    if show:
        bkp.show(ax, toolbar_location="above")

    return ax
