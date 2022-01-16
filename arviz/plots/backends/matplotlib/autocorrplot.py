"""Matplotlib Autocorrplot."""
import matplotlib.pyplot as plt
import numpy as np

from ....stats import autocorr
from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show, create_axes_grid


def plot_autocorr(
    axes,
    plotters,
    max_lag,
    figsize,
    rows,
    cols,
    combined,
    textsize,
    labeller,
    backend_kwargs,
    show,
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
    backend_kwargs.setdefault("squeeze", True)

    if axes is None:
        _, axes = create_axes_grid(
            len(plotters),
            rows,
            cols,
            backend_kwargs=backend_kwargs,
        )

    for (var_name, selection, isel, x), ax in zip(plotters, np.ravel(axes)):
        x_prime = x
        if combined:
            x_prime = x.flatten()
        c_i = 1.96 / x_prime.size ** 0.5
        y = autocorr(x_prime)

        ax.fill_between([0, max_lag], -c_i, c_i, color="0.75")
        ax.vlines(x=np.arange(0, max_lag), ymin=0, ymax=y[0:max_lag], lw=linewidth)

        ax.set_title(
            labeller.make_label_vert(var_name, selection, isel), fontsize=titlesize, wrap=True
        )
        ax.tick_params(labelsize=xt_labelsize)

    if np.asarray(axes).size > 0:
        np.asarray(axes).item(0).set_xlim(0, max_lag)
        np.asarray(axes).item(0).set_ylim(-1, 1)

    if backend_show(show):
        plt.show()

    return axes
