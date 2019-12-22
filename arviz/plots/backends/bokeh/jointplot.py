"""Bokeh jointplot."""
import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import gridplot

from . import backend_kwarg_defaults, backend_show
from ...distplot import plot_dist
from ...kdeplot import plot_kde
from ...plot_utils import make_label


def plot_joint(
    ax,
    figsize,
    plotters,
    xt_labelsize,
    kind,
    contour,
    fill_last,
    joint_kwargs,
    gridsize,
    marginal_kwargs,
    backend_kwargs,
    show,
):
    """Bokeh joint plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(
            ("tools", "plot.bokeh.tools"),
            ("output_backend", "plot.bokeh.output_backend"),
            ("dpi", "plot.bokeh.figure.dpi"),
        ),
        **backend_kwargs,
    }
    dpi = backend_kwargs.pop("dpi")
    if ax is None:
        axjoin = bkp.figure(
            width=int(figsize[0] * dpi * 0.8), height=int(figsize[1] * dpi * 0.8), **backend_kwargs
        )
        ax_hist_x = bkp.figure(
            width=int(figsize[0] * dpi * 0.8),
            height=int(figsize[1] * dpi * 0.2),
            x_range=axjoin.x_range,
            **backend_kwargs
        )
        ax_hist_y = bkp.figure(
            width=int(figsize[0] * dpi * 0.2),
            height=int(figsize[1] * dpi * 0.8),
            y_range=axjoin.y_range,
            **backend_kwargs
        )

    elif len(ax) == 2 and len(ax[0]) == 2 and len(ax[1]) == 2:
        ax_hist_x, _ = ax[0]
        axjoin, ax_hist_y = ax[1]
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
            backend_kwargs={},
            show=False,
            **joint_kwargs
        )
    else:
        if gridsize == "auto":
            gridsize = int(len(x) ** 0.35)
            gridsize = gridsize / 10

        axjoin.hexbin(x, y, size=gridsize, **joint_kwargs)

    marginal_kwargs["plot_kwargs"].setdefault("line_color", "black")
    for val, ax_, rotate in ((x, ax_hist_x, False), (y, ax_hist_y, True)):
        plot_dist(
            val,
            textsize=xt_labelsize,
            rotated=rotate,
            ax=ax_,
            backend="bokeh",
            backend_kwargs={},
            show=False,
            **marginal_kwargs
        )

    if backend_show(show):
        grid = gridplot([[ax_hist_x, None], [axjoin, ax_hist_y]], toolbar_location="above")
        bkp.show(grid)

    return np.array([[ax_hist_x, None], [axjoin, ax_hist_y]])
