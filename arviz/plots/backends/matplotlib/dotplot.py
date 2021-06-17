import matplotlib.pyplot as plt

from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, create_axes_grid
from .plotpointinterval import plot_point_interval


def plot_dots(
    values,
    binwidth,
    dotsize,
    stackratio,
    hdi_prob,
    rotated,
    dotcolor,
    intervalcolor,
    markersize,
    figsize,
    linewidth,
    point_estimate,
    quantiles,
    point_interval,
    backend_kwargs,
    plot_kwargs,
    interval_kwargs,
):

    if backend_kwargs is None:
        backend_kwargs = {}
        backend_kwargs = {
            **backend_kwarg_defaults(),
        }

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

    if interval_kwargs is None:
        interval_kwargs = {}
        interval_kwargs.setdefault("color", intervalcolor)

    _, ax = create_axes_grid(1, backend_kwargs=backend_kwargs,)

    ## Wilkinson's Algorithm
    count = 0

    while count < quantiles:
        stack_first_dot = values[count]
        num_dots_stack = 0
        while values[count] < (binwidth + stack_first_dot):
            num_dots_stack += 1
            count += 1
            if count == quantiles:
                break
        x_coord = (stack_first_dot + values[count - 1]) / 2
        y_coord = binwidth / 2
        if rotated:
            x_coord, y_coord = y_coord, x_coord
        for _ in range(num_dots_stack):
            dot = plt.Circle((x_coord, y_coord), dotsize * binwidth / 2, **plot_kwargs)
            ax.add_patch(dot)
            if rotated:
                x_coord += binwidth + (stackratio - 1) * (binwidth)
            else:
                y_coord += binwidth + (stackratio - 1) * (binwidth)

    if point_interval:
        ax = plot_point_interval(
            ax, values, point_estimate, hdi_prob, linewidth, markersize, rotated, interval_kwargs
        )

    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()

    return ax
