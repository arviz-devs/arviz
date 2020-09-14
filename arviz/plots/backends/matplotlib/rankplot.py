"""Matplotlib rankplot."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from ....stats.density_utils import histogram
from ...plot_utils import _scale_fig_size, make_label
from . import backend_kwarg_defaults, backend_show, create_axes_grid


def plot_rank(
    axes,
    length_plotters,
    rows,
    cols,
    figsize,
    plotters,
    bins,
    kind,
    colors,
    ref_line,
    labels,
    backend_kwargs,
    show,
):
    """Matplotlib rankplot.."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, ax_labelsize, titlesize, _, _, _ = _scale_fig_size(figsize, None, rows=rows, cols=cols)
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs.setdefault("squeeze", True)
    if axes is None:
        _, axes = create_axes_grid(
            length_plotters,
            rows,
            cols,
            backend_kwargs=backend_kwargs,
        )

    for ax, (var_name, selection, var_data) in zip(np.ravel(axes), plotters):
        ranks = scipy.stats.rankdata(var_data, method="average").reshape(var_data.shape)
        bin_ary = np.histogram_bin_edges(ranks, bins=bins, range=(0, ranks.size))
        all_counts = np.empty((len(ranks), len(bin_ary) - 1))
        for idx, row in enumerate(ranks):
            _, all_counts[idx], _ = histogram(row, bins=bin_ary)
        gap = all_counts.max() * 1.05
        width = bin_ary[1] - bin_ary[0]

        # Center the bins
        bin_ary = (bin_ary[1:] + bin_ary[:-1]) / 2

        y_ticks = []
        if kind == "bars":
            for idx, counts in enumerate(all_counts):
                y_ticks.append(idx * gap)
                ax.bar(
                    bin_ary,
                    counts,
                    bottom=y_ticks[-1],
                    width=width,
                    align="center",
                    color=colors[idx],
                    edgecolor=ax.get_facecolor(),
                )
                if ref_line:
                    ax.axhline(y=y_ticks[-1] + counts.mean(), linestyle="--", color="k")
            if labels:
                ax.set_ylabel("Chain", fontsize=ax_labelsize)
        elif kind == "vlines":
            ymin = all_counts.mean()
            for idx, counts in enumerate(all_counts):
                ax.plot(bin_ary, counts, "o", color=colors[idx])
                ax.vlines(bin_ary, ymin, counts, lw=2, colors=colors[idx])
            ax.set_ylim(0, all_counts.mean() * 2)
            if ref_line:
                ax.axhline(y=all_counts.mean(), linestyle="--", color="k")

        if labels:
            ax.set_xlabel("Rank (all chains)", fontsize=ax_labelsize)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(np.arange(len(y_ticks)))
            ax.set_title(make_label(var_name, selection), fontsize=titlesize)
        else:
            ax.set_yticks([])

    if backend_show(show):
        plt.show()

    return axes
