"""Matplotlib Density Comparison plot."""
import matplotlib.pyplot as plt
import numpy as np

from . import backend_show
from ...distplot import plot_dist
from ...plot_utils import make_label
from . import backend_kwarg_defaults


def plot_dist_comparison(
    ax,
    nvars,
    ngroups,
    figsize,
    dc_plotters,
    legend,
    groups,
    prior_kwargs,
    posterior_kwargs,
    observed_kwargs,
    backend_kwargs,
    show,
):
    """Matplotlib Density Comparison plot."""
    backend_kwargs = {**backend_kwarg_defaults(), **backend_kwargs}
    if ax is None:
        axes = np.empty((nvars, ngroups + 1), dtype=object)
        fig = plt.figure(**backend_kwargs, figsize=figsize)
        gs = fig.add_gridspec(ncols=ngroups, nrows=nvars * 2)
        for i in range(nvars):
            for j in range(ngroups):
                axes[i, j] = fig.add_subplot(gs[2 * i, j])
            axes[i, -1] = fig.add_subplot(gs[2 * i + 1, :])

    else:
        axes = ax
        if ax.shape != (nvars, ngroups + 1):
            raise ValueError(
                "Found {} shape of axes, which is not equal to data shape {}.".format(
                    axes.shape, (nvars, ngroups + 1)
                )
            )

    for idx, plotter in enumerate(dc_plotters):
        group = groups[idx]
        kwargs = (
            prior_kwargs
            if group.startswith("prior")
            else posterior_kwargs
            if group.startswith("posterior")
            else observed_kwargs
        )
        for idx2, (var, selection, data,) in enumerate(plotter):
            label = make_label(var, selection)
            label = f"{group} {label}"
            plot_dist(
                data, label=label if legend else None, ax=axes[idx2, idx], **kwargs,
            )
            plot_dist(
                data, label=label if legend else None, ax=axes[idx2, -1], **kwargs,
            )

    if backend_show(show):
        plt.show()

    return axes
