"""Matplotlib Autocorrplot."""
import matplotlib.pyplot as plt
import numpy as np

from ....stats import autocorr
from ...plot_utils import create_axes_grid, _scale_fig_size, make_label
from . import backend_kwarg_defaults, backend_show


def plot_autocorr(
    axes, plotters, max_lag, figsize, rows, cols, combined, textsize, backend_kwargs, show,
):
    """Matplotlib autocorrplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, _, titlesize, xt_labelsize, linewidth, _ = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs.setdefault("sharex", True)
    backend_kwargs.setdefault("sharey", True)

    if axes is None:
        _, axes = create_axes_grid(len(plotters), rows, cols, backend_kwargs=backend_kwargs,)

    for (var_name, selection, x), ax in zip(plotters, axes):
        x_prime = x
        if combined:
            x_prime = x.flatten()
        y = autocorr(x_prime)
        ax.vlines(x=np.arange(0, max_lag), ymin=0, ymax=y[0:max_lag], lw=linewidth)
        ax.hlines(0, 0, max_lag, "steelblue")
        ax.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
        ax.tick_params(labelsize=xt_labelsize)

    if axes.size > 0:
        axes[0].set_xlim(0, max_lag)
        axes[0].set_ylim(-1, 1)

    if backend_show(show):
        plt.show()

    return axes
