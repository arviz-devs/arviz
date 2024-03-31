"""Matplotlib Parallel coordinates plot."""

import matplotlib.pyplot as plt
import numpy as np

from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show, create_axes_grid


def plot_parallel(
    ax,
    colornd,
    colord,
    shadend,
    diverging_mask,
    posterior,
    textsize,
    var_names,
    legend,
    figsize,
    backend_kwargs,
    backend_config,  # pylint: disable=unused-argument
    show,
):
    """Matplotlib parallel plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, _, _, xt_labelsize, _, _ = _scale_fig_size(figsize, textsize, 1, 1)
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True
    if ax is None:
        _, ax = create_axes_grid(1, backend_kwargs=backend_kwargs)

    ax.plot(posterior[:, ~diverging_mask], color=colornd, alpha=shadend)

    if np.any(diverging_mask):
        ax.plot(posterior[:, diverging_mask], color=colord, lw=1)

    ax.tick_params(labelsize=textsize)
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names)

    if legend:
        ax.plot([], color=colornd, label="non-divergent")
        if np.any(diverging_mask):
            ax.plot([], color=colord, label="divergent")
        ax.legend(fontsize=xt_labelsize)

    if backend_show(show):
        plt.show()

    return ax
