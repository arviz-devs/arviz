"""Matplotlib Violinplot."""
import matplotlib.pyplot as plt
import numpy as np

from . import backend_show
from ....stats import hdi
from ....numeric_utils import _fast_kde, histogram, get_bins
from ...plot_utils import make_label, _create_axes_grid


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
    bw,
    hdi_prob,
    linewidth,
    ax_labelsize,
    xt_labelsize,
    quartiles,
    backend_kwargs,
    show,
):
    """Matplotlib violin plot."""
    if ax is None:
        fig, ax = _create_axes_grid(
            len(plotters),
            rows,
            cols,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            squeeze=False,
            backend_kwargs=backend_kwargs,
        )
        fig.set_constrained_layout(False)
        fig.subplots_adjust(wspace=0)

    ax = np.atleast_1d(ax)

    for (var_name, selection, x), ax_ in zip(plotters, ax.flatten()):
        val = x.flatten()
        if val[0].dtype.kind == "i":
            dens = cat_hist(val, rug, shade, ax_, **shade_kwargs)
        else:
            dens = _violinplot(val, rug, shade, bw, ax_, **shade_kwargs)

        if rug:
            rug_x = -np.abs(np.random.normal(scale=max(dens) / 3.5, size=len(val)))
            ax_.plot(rug_x, val, **rug_kwargs)

        per = np.percentile(val, [25, 75, 50])
        hdi_probs = hdi(val, hdi_prob, multimodal=False)

        if quartiles:
            ax_.plot([0, 0], per[:2], lw=linewidth * 3, color="k", solid_capstyle="round")
        ax_.plot([0, 0], hdi_probs, lw=linewidth, color="k", solid_capstyle="round")
        ax_.plot(0, per[-1], "wo", ms=linewidth * 1.5)

        ax_.set_xlabel(make_label(var_name, selection), fontsize=ax_labelsize)
        ax_.set_xticks([])
        ax_.tick_params(labelsize=xt_labelsize)
        ax_.grid(None, axis="x")

    if backend_show(show):
        plt.show()

    return ax


def _violinplot(val, rug, shade, bw, ax, **shade_kwargs):
    """Auxiliary function to plot violinplots."""
    density, low_b, up_b = _fast_kde(val, bw=bw)
    x = np.linspace(low_b, up_b, len(density))

    if not rug:
        x = np.concatenate([x, x[::-1]])
        density = np.concatenate([-density, density[::-1]])

    ax.fill_betweenx(x, density, alpha=shade, lw=0, **shade_kwargs)
    return density


def cat_hist(val, rug, shade, ax, **shade_kwargs):
    """Auxiliary function to plot discrete-violinplots."""
    bins = get_bins(val)
    _, binned_d, _ = histogram(val, bins=bins)

    bin_edges = np.linspace(np.min(val), np.max(val), len(bins))
    heights = np.diff(bin_edges)
    centers = bin_edges[:-1] + heights.mean() / 2

    if rug:
        left = None
    else:
        left = -0.5 * binned_d

    ax.barh(centers, binned_d, height=heights, left=left, alpha=shade, **shade_kwargs)
    return binned_d
