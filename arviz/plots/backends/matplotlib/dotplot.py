import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers

from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, create_axes_grid, backend_show
from ...plot_utils import plot_point_interval


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

    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {**backend_kwarg_defaults(), **backend_kwargs}

    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True

    (figsize, _, _, _, auto_linewidth, auto_markersize) = _scale_fig_size(figsize, None)

    if plot_kwargs is None:
        plot_kwargs = {}
        plot_kwargs.setdefault("color", dotcolor)

    if linewidth is None:
        linewidth = auto_linewidth

    if markersize is None:
        markersize = auto_markersize

    if ax is None:
        fig_manager = _pylab_helpers.Gcf.get_active()
        if fig_manager is not None:
            ax = fig_manager.canvas.figure.gca()
        else:
            _, ax = create_axes_grid(
                1,
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
            "matplotlib",
        )

    if nquantiles > values.shape[0]:
        nquantiles = values.shape[0]
    else:
        qlist = np.linspace(1 / (2 * nquantiles), 1 - 1 / (2 * nquantiles), nquantiles)
        values = np.quantile(values, qlist)

    if binwidth is None:
        binwidth = math.sqrt((values[-1] - values[0] + 1) ** 2 / (2 * nquantiles * np.pi))

    ## Wilkinson's Algorithm
    x, y = wilkinson_algorithm(values, nquantiles, binwidth, stackratio, rotated)

    for (x_i, y_i) in zip(x, y):
        dot = plt.Circle((x_i, y_i), dotsize * binwidth / 2, **plot_kwargs)
        ax.add_patch(dot)

    if rotated:
        ax.tick_params(bottom=False, labelbottom=False)
    else:
        ax.tick_params(left=False, labelleft=False)

    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()

    if backend_show(show):
        plt.show()

    return ax


def wilkinson_algorithm(values, nquantiles, binwidth, stackratio, rotated):
    """Uses wilkinson's algorithm to distribute dots into horizontal stacks"""

    count = 0
    x, y = [], []

    while count < nquantiles:
        stack_first_dot = values[count]
        num_dots_stack = 0
        while values[count] < (binwidth + stack_first_dot):
            num_dots_stack += 1
            count += 1
            if count == nquantiles:
                break
        x_coord = (stack_first_dot + values[count - 1]) / 2
        y_coord = binwidth / 2
        if rotated:
            x_coord, y_coord = y_coord, x_coord
        x.append(x_coord)
        y.append(y_coord)
        for _ in range(num_dots_stack):
            x.append(x_coord)
            y.append(y_coord)
            if rotated:
                x_coord += binwidth + (stackratio - 1) * (binwidth)
            else:
                y_coord += binwidth + (stackratio - 1) * (binwidth)

    return x, y
