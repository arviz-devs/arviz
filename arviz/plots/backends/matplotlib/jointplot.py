"""Matplotlib jointplot."""
import matplotlib.pyplot as plt
import numpy as np

from ...distplot import plot_dist
from ...kdeplot import plot_kde
from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show, matplotlib_kwarg_dealiaser
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
    """Matplotlib joint plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, ax_labelsize, _, xt_labelsize, linewidth, _ = _scale_fig_size(figsize, textsize)

    backend_kwargs.setdefault("figsize", figsize)

    if kind == "kde":
        types = "plot"
    elif kind == "scatter":
        types = "scatter"
    else:
        types = "hexbin"
    joint_kwargs = matplotlib_kwarg_dealiaser(joint_kwargs, types)

    if marginal_kwargs is None:
        marginal_kwargs = {}
    marginal_kwargs.setdefault("plot_kwargs", {})
    marginal_kwargs["plot_kwargs"].setdefault("linewidth", linewidth)

    if ax is None:
        # Instantiate figure and grid
        fig = plt.figure(**backend_kwargs)
        grid = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1, figure=fig)

        # Set up main plot
        axjoin = fig.add_subplot(grid[1:, :-1])
        # Set up top KDE
        ax_hist_x = fig.add_subplot(grid[0, :-1], sharex=axjoin)
        # Set up right KDE
        ax_hist_y = fig.add_subplot(grid[1:, -1], sharey=axjoin)
    elif len(ax) == 3:
        axjoin, ax_hist_x, ax_hist_y = ax
    else:
        raise ValueError(f"ax must be of length 3 but found {len(ax)}")

    # Personalize axes
    ax_hist_x.tick_params(labelleft=False, labelbottom=False)
    ax_hist_y.tick_params(labelleft=False, labelbottom=False)

    # Set labels for axes
    x_var_name = make_label(plotters[0][0], plotters[0][1])
    y_var_name = make_label(plotters[1][0], plotters[1][1])

    axjoin.set_xlabel(x_var_name, fontsize=ax_labelsize)
    axjoin.set_ylabel(y_var_name, fontsize=ax_labelsize)
    axjoin.tick_params(labelsize=xt_labelsize)

    # Flatten data
    x = plotters[0][-1].flatten()
    y = plotters[1][-1].flatten()

    if kind == "scatter":
        axjoin.scatter(x, y, **joint_kwargs)
    elif kind == "kde":
        plot_kde(x, y, contour=contour, fill_last=fill_last, ax=axjoin, **joint_kwargs)
    else:
        if gridsize == "auto":
            gridsize = int(len(x) ** 0.35)
        axjoin.hexbin(x, y, mincnt=1, gridsize=gridsize, **joint_kwargs)
        axjoin.grid(False)

    for val, ax_, rotate in ((x, ax_hist_x, False), (y, ax_hist_y, True)):
        plot_dist(val, textsize=xt_labelsize, rotated=rotate, ax=ax_, **marginal_kwargs)

    ax_hist_x.set_xlim(axjoin.get_xlim())
    ax_hist_y.set_ylim(axjoin.get_ylim())

    if backend_show(show):
        plt.show()

    return np.array([axjoin, ax_hist_x, ax_hist_y])
