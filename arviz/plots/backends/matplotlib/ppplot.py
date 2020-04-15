"""Matplotib Posterior predictive plot."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from . import backend_show
from ...kdeplot import plot_kde
from ...plot_utils import _create_axes_grid

colors = {
    "prior": "red",
    "posterior": "blue",
    "observed": "yellow",
}


def plot_pp(
    ax,
    length_plotters,
    rows,
    cols,
    figsize,
    pp_plotters,
    linewidth,
    legend,
    groups,
    fill_kwargs,
    plot_kwargs,
    backend_kwargs,
    show,
):
    """Matplotlib pp plot."""
    if ax is None:
        fig, ax = _create_axes_grid(
            rows * cols, rows, cols, figsize=figsize, backend_kwargs=backend_kwargs
        )
        axes = []
        for i in range(cols, rows * cols, 2 * cols):
            axes += list(ax[i - cols : i])
            gs = ax[i].get_gridspec()
            for j in range(i, i + cols):
                ax[j].remove()
            axbig = fig.add_subplot(gs[i // cols, :])
            axes.append(axbig)

    else:
        axes = np.ravel(ax)
        if len(axes) != length_plotters:
            raise ValueError(
                "Found {} variables to plot but {} axes instances. They must be equal.".format(
                    length_plotters, len(axes)
                )
            )

    matplotlib.rc("xtick", labelsize=7)
    matplotlib.rc("ytick", labelsize=7)

    plot_kwargs.pop("color", None)
    plot_kwargs.setdefault("linewidth", linewidth)

    for idx, plotter in enumerate(pp_plotters):
        group = groups[idx]
        color = colors[group]
        for idx2, (var, coord, data,) in enumerate(plotter):
            _pp_helper(
                ax=axes[(cols + 1) * idx2 + idx],
                data=data,
                group=group,
                var_name=var,
                coord_name=coord,
                color=color,
                fill_kwargs=fill_kwargs,
                plot_kwargs=plot_kwargs,
            )
            _pp_helper(
                ax=axes[(cols + 1) * (idx2) + cols],
                data=data,
                group=group,
                var_name=var,
                coord_name=coord,
                color=color,
                fill_kwargs=fill_kwargs,
                plot_kwargs=plot_kwargs,
            )

    if backend_show(show):
        plt.show()

    return axes


def _pp_helper(
    ax, data, group, var_name, coord_name, color, fill_kwargs, plot_kwargs,
):
    plot_kwargs["color"] = color
    plot_kde(
        data,
        label="{} {} {}".format(group, var_name, coord_name),
        fill_kwargs=fill_kwargs,
        plot_kwargs=plot_kwargs,
        ax=ax,
    )
