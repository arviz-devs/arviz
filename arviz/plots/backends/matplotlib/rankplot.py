"""Matplotlib rankplot."""
import matplotlib.pyplot as plt
import numpy as np

from ....stats.density_utils import histogram
from ...plot_utils import _scale_fig_size, compute_ranks
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
    labeller,
    ref_line_kwargs,
    bar_kwargs,
    vlines_kwargs,
    marker_vlines_kwargs,
    backend_kwargs,
    show,
):
    """Matplotlib rankplot.."""
    if ref_line_kwargs is None:
        ref_line_kwargs = {}
    ref_line_kwargs.setdefault("linestyle", "--")
    ref_line_kwargs.setdefault("color", "k")

    if bar_kwargs is None:
        bar_kwargs = {}
    bar_kwargs.setdefault("align", "center")

    if vlines_kwargs is None:
        vlines_kwargs = {}
    vlines_kwargs.setdefault("lw", 2)

    if marker_vlines_kwargs is None:
        marker_vlines_kwargs = {}
        marker_vlines_kwargs.setdefault("marker", "o")
        marker_vlines_kwargs.setdefault("lw", 0)

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

    for ax, (var_name, selection, isel, var_data) in zip(np.ravel(axes), plotters):
        ranks = compute_ranks(var_data)
        bin_ary = np.histogram_bin_edges(ranks, bins=bins, range=(0, ranks.size))
        all_counts = np.empty((len(ranks), len(bin_ary) - 1))
        for idx, row in enumerate(ranks):
            _, all_counts[idx], _ = histogram(row, bins=bin_ary)
        gap = 2 / ranks.size
        width = bin_ary[1] - bin_ary[0]

        bar_kwargs.setdefault("width", width)
        bar_kwargs.setdefault("edgecolor", ax.get_facecolor())
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
                    color=colors[idx],
                    **bar_kwargs,
                )
                if ref_line:
                    ax.axhline(y=y_ticks[-1] + counts.mean(), **ref_line_kwargs)
            if labels:
                ax.set_ylabel("Chain", fontsize=ax_labelsize)
        elif kind == "vlines":
            ymin = all_counts.mean()

            for idx, counts in enumerate(all_counts):
                ax.plot(bin_ary, counts, color=colors[idx], **marker_vlines_kwargs)
                ax.vlines(bin_ary, ymin, counts, colors=colors[idx], **vlines_kwargs)
            ax.set_ylim(0, all_counts.mean() * 2)
            if ref_line:
                ax.axhline(y=ymin, **ref_line_kwargs)

        if labels:
            ax.set_xlabel("Rank (all chains)", fontsize=ax_labelsize)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(np.arange(len(y_ticks)))
            ax.set_title(labeller.make_label_vert(var_name, selection, isel), fontsize=titlesize)
        else:
            ax.set_yticks([])

    if backend_show(show):
        plt.show()

    return axes
