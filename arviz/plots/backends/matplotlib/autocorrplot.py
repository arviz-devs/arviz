"""Matplotlib Autocorrplot."""
import matplotlib.pyplot as plt
import numpy as np

from . import backend_show
from ....stats import autocorr
from ...plot_utils import _create_axes_grid, make_label


def plot_autocorr(
    axes,
    plotters,
    max_lag,
    figsize,
    rows,
    cols,
    linewidth,
    titlesize,
    combined,
    xt_labelsize,
    backend_kwargs,
    show,
):
    """Matplotlib autocorrplot."""
    if axes is None:
        _, axes = _create_axes_grid(
            len(plotters),
            rows,
            cols,
            figsize=figsize,
            squeeze=False,
            sharex=True,
            sharey=True,
            backend="matplotlib",
            backend_kwargs=backend_kwargs,
        )

    axes = np.atleast_2d(axes)  # in case of only 1 plot

    for (var_name, selection, x), ax in zip(plotters, axes.flatten()):
        x_prime = x
        if combined:
            x_prime = x.flatten()
        y = autocorr(x_prime)
        ax.vlines(x=np.arange(0, max_lag), ymin=0, ymax=y[0:max_lag], lw=linewidth)
        ax.hlines(0, 0, max_lag, "steelblue")
        ax.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
        ax.tick_params(labelsize=xt_labelsize)

    if axes.size > 0:
        axes[0, 0].set_xlim(0, max_lag)
        axes[0, 0].set_ylim(-1, 1)

    if backend_show(show):
        plt.show()

    return axes
