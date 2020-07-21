"""Matplotlib Density Comparison plot."""
import matplotlib.pyplot as plt
import numpy as np

from . import backend_show
from ...distplot import plot_dist
from ...plot_utils import _scale_fig_size, make_label, vectorized_to_hex
from . import backend_kwarg_defaults


def plot_dist_comparison(
    ax,
    nvars,
    ngroups,
    figsize,
    dc_plotters,
    legend,
    groups,
    textsize,
    prior_kwargs,
    posterior_kwargs,
    observed_kwargs,
    backend_kwargs,
    show,
):
    """Matplotlib Density Comparison plot."""
    if prior_kwargs is None:
        prior_kwargs = {}

    if posterior_kwargs is None:
        posterior_kwargs = {}

    if observed_kwargs is None:
        observed_kwargs = {}

    if backend_kwargs is None:
        backend_kwargs = {}

    (figsize, _, _, _, linewidth, _) = _scale_fig_size(figsize, textsize, 2 * nvars, ngroups)

    posterior_kwargs.setdefault("plot_kwargs", dict())
    posterior_kwargs["plot_kwargs"]["color"] = vectorized_to_hex(
        posterior_kwargs["plot_kwargs"].get("color", "C0")
    )
    posterior_kwargs["plot_kwargs"].setdefault("linewidth", linewidth)
    posterior_kwargs.setdefault("hist_kwargs", dict())
    posterior_kwargs["hist_kwargs"].setdefault("alpha", 0.5)

    prior_kwargs.setdefault("plot_kwargs", dict())
    prior_kwargs["plot_kwargs"]["color"] = vectorized_to_hex(
        prior_kwargs["plot_kwargs"].get("color", "C1")
    )
    prior_kwargs["plot_kwargs"].setdefault("linewidth", linewidth)
    prior_kwargs.setdefault("hist_kwargs", dict())
    prior_kwargs["hist_kwargs"].setdefault("alpha", 0.5)

    observed_kwargs.setdefault("plot_kwargs", dict())
    observed_kwargs["plot_kwargs"]["color"] = vectorized_to_hex(
        observed_kwargs["plot_kwargs"].get("color", "C2")
    )
    observed_kwargs["plot_kwargs"].setdefault("linewidth", linewidth)
    observed_kwargs.setdefault("hist_kwargs", dict())
    observed_kwargs["hist_kwargs"].setdefault("alpha", 0.5)

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
