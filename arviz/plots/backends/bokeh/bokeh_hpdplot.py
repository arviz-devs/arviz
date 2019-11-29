"""Bokeh hpdplot."""
from itertools import cycle

import bokeh.plotting as bkp
from matplotlib.pyplot import rcParams
import numpy as np


def _plot_hpdplot(ax, x_data, y_data, plot_kwargs, fill_kwargs, show):
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
        ax = bkp.figure(width=500, height=300, output_backend="webgl", tools=tools)

    color = plot_kwargs.pop("color")
    if len(color) == 2 and color[0] == "C":
        color = [
            prop
            for _, prop in zip(
                range(int(color[1:])), cycle(rcParams["axes.prop_cycle"].by_key()["color"])
            )
        ][-1]
    plot_kwargs.setdefault("line_color", color)
    plot_kwargs.setdefault("line_alpha", plot_kwargs.pop("alpha", 0))

    color = fill_kwargs.pop("color")
    if len(color) == 2 and color[0] == "C":
        color = [
            prop
            for _, prop in zip(
                range(int(color[1:])), cycle(rcParams["axes.prop_cycle"].by_key()["color"])
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
