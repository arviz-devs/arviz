"""Matplotlib Density Comparison plot."""

import matplotlib.pyplot as plt
import numpy as np

from ...distplot import plot_dist
from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show


def plot_dist_comparison(
    ax,
    nvars,
    ngroups,
    figsize,
    dc_plotters,
    legend,
    groups,
    textsize,
    labeller,
    prior_kwargs,
    posterior_kwargs,
    observed_kwargs,
    backend_kwargs,
    show,
):
    """Matplotlib Density Comparison plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    if prior_kwargs is None:
        prior_kwargs = {}

    if posterior_kwargs is None:
        posterior_kwargs = {}

    if observed_kwargs is None:
        observed_kwargs = {}

    if backend_kwargs is None:
        backend_kwargs = {}

    (figsize, _, _, _, linewidth, _) = _scale_fig_size(figsize, textsize, 2 * nvars, ngroups)

    backend_kwargs.setdefault("figsize", figsize)

    posterior_kwargs.setdefault("plot_kwargs", {})
    posterior_kwargs["plot_kwargs"]["color"] = posterior_kwargs["plot_kwargs"].get("color", "C0")
    posterior_kwargs["plot_kwargs"].setdefault("linewidth", linewidth)
    posterior_kwargs.setdefault("hist_kwargs", {})
    posterior_kwargs["hist_kwargs"].setdefault("alpha", 0.5)

    prior_kwargs.setdefault("plot_kwargs", {})
    prior_kwargs["plot_kwargs"]["color"] = prior_kwargs["plot_kwargs"].get("color", "C1")
    prior_kwargs["plot_kwargs"].setdefault("linewidth", linewidth)
    prior_kwargs.setdefault("hist_kwargs", {})
    prior_kwargs["hist_kwargs"].setdefault("alpha", 0.5)

    observed_kwargs.setdefault("plot_kwargs", {})
    observed_kwargs["plot_kwargs"]["color"] = observed_kwargs["plot_kwargs"].get("color", "C2")
    observed_kwargs["plot_kwargs"].setdefault("linewidth", linewidth)
    observed_kwargs.setdefault("hist_kwargs", {})
    observed_kwargs["hist_kwargs"].setdefault("alpha", 0.5)

    if ax is None:
        axes = np.empty((nvars, ngroups + 1), dtype=object)
        fig = plt.figure(**backend_kwargs)
        gs = fig.add_gridspec(ncols=ngroups, nrows=nvars * 2)
        for i in range(nvars):
            for j in range(ngroups):
                axes[i, j] = fig.add_subplot(gs[2 * i, j])
            axes[i, -1] = fig.add_subplot(gs[2 * i + 1, :])

    else:
        axes = ax
        if ax.shape != (nvars, ngroups + 1):
            raise ValueError(
                f"Found {axes.shape} shape of axes, "
                f"which is not equal to data shape {(nvars, ngroups + 1)}."
            )

    for idx, plotter in enumerate(dc_plotters):
        group = groups[idx]
        kwargs = (
            prior_kwargs
            if group.startswith("prior")
            else posterior_kwargs if group.startswith("posterior") else observed_kwargs
        )
        for idx2, (
            var_name,
            sel,
            isel,
            data,
        ) in enumerate(plotter):
            label = f"{group}"
            plot_dist(
                data,
                label=label if legend else None,
                ax=axes[idx2, idx],
                **kwargs,
            )
            plot_dist(
                data,
                label=label if legend else None,
                ax=axes[idx2, -1],
                **kwargs,
            )
            if idx == 0:
                axes[idx2, -1].set_xlabel(labeller.make_label_vert(var_name, sel, isel))

    if backend_show(show):
        plt.show()

    return axes
