"""Matplotlib jointplot."""
import matplotlib.pyplot as plt
import numpy as np

from . import backend_kwarg_defaults, backend_show
from ...distplot import plot_dist
from ...kdeplot import plot_kde
from ...plot_utils import make_label


def plot_joint(
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
    if ax is None:
        # Instantiate figure and grid
        fig, _ = plt.subplots(0, 0, figsize=figsize, **backend_kwargs)
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
        raise ValueError("ax must be of lenght 3 but found {}".format(len(ax)))

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
    x = plotters[0][2].flatten()
    y = plotters[1][2].flatten()

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
