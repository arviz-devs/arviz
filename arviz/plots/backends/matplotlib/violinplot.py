"""Matplotlib Violinplot."""
import matplotlib.pyplot as plt
import numpy as np

from . import backend_show
from ....stats import hpd
from ....stats.stats_utils import histogram
from ...kdeplot import _fast_kde
from ...plot_utils import get_bins, make_label, _create_axes_grid


def plot_violin(
    ax,
    plotters,
    figsize,
    rows,
    cols,
    sharey,
    kwargs_shade,
    shade,
    bw,
    credible_interval,
    linewidth,
    ax_labelsize,
    xt_labelsize,
    quartiles,
    backend_kwargs,
    show,
):
    """Matplotlib violin plot."""
    if ax is None:
        _, ax = _create_axes_grid(
            len(plotters),
            rows,
            cols,
            sharey=sharey,
            figsize=figsize,
            squeeze=False,
            backend_kwargs=backend_kwargs,
        )

    ax = np.atleast_1d(ax)

    for (var_name, selection, x), ax_ in zip(plotters, ax.flatten()):
        val = x.flatten()
        if val[0].dtype.kind == "i":
            cat_hist(val, shade, ax_, **kwargs_shade)
        else:
            _violinplot(val, shade, bw, ax_, **kwargs_shade)

        per = np.percentile(val, [25, 75, 50])
        hpd_intervals = hpd(val, credible_interval, multimodal=False)

        if quartiles:
            ax_.plot([0, 0], per[:2], lw=linewidth * 3, color="k", solid_capstyle="round")
        ax_.plot([0, 0], hpd_intervals, lw=linewidth, color="k", solid_capstyle="round")
        ax_.plot(0, per[-1], "wo", ms=linewidth * 1.5)

        ax_.set_xlabel(make_label(var_name, selection), fontsize=ax_labelsize)
        ax_.set_xticks([])
        ax_.tick_params(labelsize=xt_labelsize)
        ax_.grid(None, axis="x")

    if backend_show(show):
        plt.show()

    return ax


def _violinplot(val, shade, bw, ax, **kwargs_shade):
    """Auxiliary function to plot violinplots."""
    density, low_b, up_b = _fast_kde(val, bw=bw)
    x = np.linspace(low_b, up_b, len(density))

    x = np.concatenate([x, x[::-1]])
    density = np.concatenate([-density, density[::-1]])

    ax.fill_betweenx(x, density, alpha=shade, lw=0, **kwargs_shade)


def cat_hist(val, shade, ax, **kwargs_shade):
    """Auxiliary function to plot discrete-violinplots."""
    bins = get_bins(val)
    _, binned_d, _ = histogram(val, bins=bins)

    bin_edges = np.linspace(np.min(val), np.max(val), len(bins))
    centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
    heights = np.diff(bin_edges)

    lefts = -0.5 * binned_d
    ax.barh(centers, binned_d, height=heights, left=lefts, alpha=shade, **kwargs_shade)
