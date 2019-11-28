"""Matplotlib jointplot."""
import bokeh.plotting as bkp
from bokeh.layouts import gridplot
import matplotlib.pyplot as plt
import numpy as np

from ...distplot import plot_dist
from ...kdeplot import plot_kde
from ...plot_utils import make_label


def _plot_joint(
    ax,
    figsize,
    plotters,
    ax_labelsize,
    xt_labelsize,
    kind,
    contour,
    fill_last,
    joint_kwargs,
    gridsize,
    marginal_kwargs,
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
        axjoin = bkp.figure(
            width=int(figsize[0] * 90 * 0.8),
            height=int(figsize[1] * 90 * 0.8),
            output_backend="webgl",
            tools=tools,
        )
        ax_hist_x = bkp.figure(
            width=int(figsize[0] * 90 * 0.8),
            height=int(figsize[1] * 90 * 0.2),
            output_backend="webgl",
            tools=tools,
            x_range=axjoin.x_range,
        )
        ax_hist_y = bkp.figure(
            width=int(figsize[0] * 90 * 0.2),
            height=int(figsize[1] * 90 * 0.8),
            output_backend="webgl",
            tools=tools,
            y_range=axjoin.y_range,
        )

    elif len(ax) == 3:
        axjoin, ax_hist_x, ax_hist_y = ax
    else:
        raise ValueError("ax must be of lenght 3 but found {}".format(len(ax)))

    # Set labels for axes
    x_var_name = make_label(plotters[0][0], plotters[0][1])
    y_var_name = make_label(plotters[1][0], plotters[1][1])

    axjoin.xaxis.axis_label = x_var_name
    axjoin.yaxis.axis_label = y_var_name

    # Flatten data
    x = plotters[0][2].flatten()
    y = plotters[1][2].flatten()

    if kind == "scatter":
        axjoin.circle(x, y, **joint_kwargs)
    elif kind == "kde":
        plot_kde(
            x,
            y,
            contour=contour,
            fill_last=fill_last,
            ax=axjoin,
            backend="bokeh",
            show=False,
            **joint_kwargs
        )
    else:
        if gridsize == "auto":
            gridsize = int(len(x) ** 0.35)

        axjoin.hexbin(x, y, size=gridsize, **joint_kwargs)

    marginal_kwargs["plot_kwargs"].setdefault("line_color", "black")
    for val, ax_, rotate in ((x, ax_hist_x, False), (y, ax_hist_y, True)):
        plot_dist(
            val,
            textsize=xt_labelsize,
            rotated=rotate,
            ax=ax_,
            backend="bokeh",
            show=False,
            **marginal_kwargs
        )

    if show:
        grid = gridplot([[ax_hist_x, None], [axjoin, ax_hist_y]], toolbar_location="above")
        bkp.show(grid)

    return np.array([[ax_hist_x, None], [axjoin, ax_hist_y]])
