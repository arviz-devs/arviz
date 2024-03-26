"""Matplotlib Violinplot."""

import matplotlib.pyplot as plt
import numpy as np

from ....stats import hdi
from ....stats.density_utils import get_bins, histogram, kde
from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show, create_axes_grid, matplotlib_kwarg_dealiaser


def plot_violin(
    ax,
    plotters,
    figsize,
    rows,
    cols,
    sharex,
    sharey,
    shade_kwargs,
    shade,
    rug,
    rug_kwargs,
    side,
    bw,
    textsize,
    labeller,
    circular,
    hdi_prob,
    quartiles,
    backend_kwargs,
    show,
):
    """Matplotlib violin plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, _) = _scale_fig_size(
        figsize, textsize, rows, cols
    )
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs.setdefault("sharex", sharex)
    backend_kwargs.setdefault("sharey", sharey)
    backend_kwargs.setdefault("squeeze", True)

    shade_kwargs = matplotlib_kwarg_dealiaser(shade_kwargs, "hexbin")
    rug_kwargs = matplotlib_kwarg_dealiaser(rug_kwargs, "plot")
    rug_kwargs.setdefault("alpha", 0.1)
    rug_kwargs.setdefault("marker", ".")
    rug_kwargs.setdefault("linestyle", "")

    if ax is None:
        fig, ax = create_axes_grid(
            len(plotters),
            rows,
            cols,
            backend_kwargs=backend_kwargs,
        )
        fig.set_layout_engine("none")
        fig.subplots_adjust(wspace=0)

    ax = np.atleast_1d(ax)

    current_col = 0
    for (var_name, selection, isel, x), ax_ in zip(plotters, ax.flatten()):
        val = x.flatten()
        if val[0].dtype.kind == "i":
            dens = cat_hist(val, rug, side, shade, ax_, **shade_kwargs)
        else:
            dens = _violinplot(val, rug, side, shade, bw, circular, ax_, **shade_kwargs)

        if rug:
            rug_x = -np.abs(np.random.normal(scale=max(dens) / 3.5, size=len(val)))
            ax_.plot(rug_x, val, **rug_kwargs)

        per = np.nanpercentile(val, [25, 75, 50])
        hdi_probs = hdi(val, hdi_prob, multimodal=False, skipna=True)

        if quartiles:
            ax_.plot([0, 0], per[:2], lw=linewidth * 3, color="k", solid_capstyle="round")
        ax_.plot([0, 0], hdi_probs, lw=linewidth, color="k", solid_capstyle="round")
        ax_.plot(0, per[-1], "wo", ms=linewidth * 1.5)

        ax_.set_title(labeller.make_label_vert(var_name, selection, isel), fontsize=ax_labelsize)
        ax_.set_xticks([])
        ax_.tick_params(labelsize=xt_labelsize)
        ax_.grid(None, axis="x")
        if current_col != 0:
            ax_.spines["left"].set_visible(False)
            ax_.yaxis.set_ticks_position("none")
        current_col += 1
        if current_col == cols:
            current_col = 0

    if backend_show(show):
        plt.show()

    return ax


def _violinplot(val, rug, side, shade, bw, circular, ax, **shade_kwargs):
    """Auxiliary function to plot violinplots."""
    if bw == "default":
        bw = "taylor" if circular else "experimental"
    x, density = kde(val, circular=circular, bw=bw)

    if rug and side == "both":
        side = "right"

    if side == "left":
        dens = -density
    elif side == "right":
        x = x[::-1]
        dens = density[::-1]
    elif side == "both":
        x = np.concatenate([x, x[::-1]])
        dens = np.concatenate([-density, density[::-1]])

    ax.fill_betweenx(x, dens, alpha=shade, lw=0, **shade_kwargs)
    return density


def cat_hist(val, rug, side, shade, ax, **shade_kwargs):
    """Auxiliary function to plot discrete-violinplots."""
    bins = get_bins(val)
    _, binned_d, _ = histogram(val, bins=bins)

    bin_edges = np.linspace(np.min(val), np.max(val), len(bins))
    heights = np.diff(bin_edges)
    centers = bin_edges[:-1] + heights.mean() / 2

    if rug and side == "both":
        side = "right"

    if side == "right":
        left = None
    elif side == "left":
        left = -binned_d
    elif side == "both":
        left = -0.5 * binned_d

    ax.barh(centers, binned_d, height=heights, left=left, alpha=shade, **shade_kwargs)
    return binned_d
