"""Bokeh jointplot."""
import bokeh.plotting as bkp
import numpy as np

from ...distplot import plot_dist
from ...kdeplot import plot_kde
from .. import show_layout
from . import backend_kwarg_defaults
from ...plot_utils import _scale_fig_size
from ....sel_utils import make_label


def plot_joint(
    ax,
    figsize,
    plotters,
    kind,
    contour,
    fill_last,
    joint_kwargs,
    gridsize,
    textsize,
    marginal_kwargs,
    backend_kwargs,
    show,
):
    """Bokeh joint plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(
            ("dpi", "plot.bokeh.figure.dpi"),
        ),
        **backend_kwargs,
    }
    dpi = backend_kwargs.pop("dpi")

    figsize, *_, xt_labelsize, linewidth, _ = _scale_fig_size(figsize, textsize)

    joint_kwargs = {} if joint_kwargs is None else joint_kwargs

    if marginal_kwargs is None:
        marginal_kwargs = {}
    marginal_kwargs.setdefault("plot_kwargs", {})
    marginal_kwargs["plot_kwargs"].setdefault("line_width", linewidth)

    if ax is None:

        backend_kwargs_join = backend_kwargs.copy()
        backend_kwargs_join.setdefault("width", int(figsize[0] * dpi * 0.8))
        backend_kwargs_join.setdefault("height", int(figsize[1] * dpi * 0.8))

        backend_kwargs_hist_x = backend_kwargs.copy()
        backend_kwargs_hist_x.setdefault("width", int(figsize[0] * dpi * 0.8))
        backend_kwargs_hist_x.setdefault("height", int(figsize[1] * dpi * 0.2))

        backend_kwargs_hist_y = backend_kwargs.copy()
        backend_kwargs_hist_y.setdefault("width", int(figsize[0] * dpi * 0.2))
        backend_kwargs_hist_y.setdefault("height", int(figsize[1] * dpi * 0.8))

        axjoin = bkp.figure(**backend_kwargs_join)

        backend_kwargs_hist_x["x_range"] = axjoin.x_range
        backend_kwargs_hist_y["y_range"] = axjoin.y_range

        ax_hist_x = bkp.figure(**backend_kwargs_hist_x)
        ax_hist_y = bkp.figure(**backend_kwargs_hist_y)

    elif len(ax) == 2 and len(ax[0]) == 2 and len(ax[1]) == 2:
        ax_hist_x, _ = ax[0]
        axjoin, ax_hist_y = ax[1]
    else:
        raise ValueError(f"ax must be of length 3 but found {len(ax)}")

    # Set labels for axes
    x_var_name = make_label(plotters[0][0], plotters[0][1])
    y_var_name = make_label(plotters[1][0], plotters[1][1])

    axjoin.xaxis.axis_label = x_var_name
    axjoin.yaxis.axis_label = y_var_name

    # Flatten data
    x = plotters[0][-1].flatten()
    y = plotters[1][-1].flatten()

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
            **joint_kwargs,
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
            **marginal_kwargs,
        )

    show_layout([[ax_hist_x, None], [axjoin, ax_hist_y]], show, force_layout=True)

    return np.array([[ax_hist_x, None], [axjoin, ax_hist_y]])
