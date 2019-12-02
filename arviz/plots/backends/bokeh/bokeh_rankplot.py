"""Bokeh rankplot."""
from itertools import cycle

import bokeh.plotting as bkp
from bokeh.models import Span
from bokeh.layouts import gridplot
import numpy as np
import scipy.stats

from ....data import convert_to_dataset
from ...plot_utils import (
    _scale_fig_size,
    xarray_var_iter,
    default_grid,
    _create_axes_grid,
    make_label,
    filter_plotters_list,
    _sturges_formula,
)
from ....utils import _var_names
from ....stats.stats_utils import histogram


def _plot_rank(
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
    ax_labelsize,
    titlesize,
    show,
):

    if axes is None:
        _, axes = _create_axes_grid(length_plotters, rows, cols, figsize=figsize, squeeze=False, backend="bokeh")

    for ax, (var_name, selection, var_data) in zip(np.ravel(axes), plotters):
        ranks = scipy.stats.rankdata(var_data).reshape(var_data.shape)
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
            bottom = 0
            for idx, counts in enumerate(all_counts):
                ax.vbar(
                    x=bin_ary,
                    top=bottom+counts,
                    bottom=bottom,
                    width=width,
                    fill_color=colors[idx],
                )
                if ref_line:
                    hline = Span(location=bottom + counts.mean(), line_dash="dashed", line_color="black")
                    ax.add_layout(hline)
                bottom += max(counts) + idx * gap
            if labels:
                ax.yaxis.axis_label = "Chain"
        elif kind == "vlines":
            ymin = np.full(len(all_counts), all_counts.mean())
            for idx, counts in enumerate(all_counts):
                ax.plot(bin_ary, counts, "o", color=colors[idx])
                ax.vlines(bin_ary, ymin, counts, lw=2, color=colors[idx])
            ax.set_ylim(0, all_counts.mean() * 2)
            if ref_line:
                ax.axhline(y=all_counts.mean(), linestyle="--", color="k")

        if labels:
            ax.xaxis.axis_label = "Rank (all chains)"
            #ax.set_yticks(y_ticks)
            #ax.set_yticklabels(np.arange(len(y_ticks)))
            #ax.set_title(make_label(var_name, selection), fontsize=titlesize)
        else:
            pass
            ax.set_yticks([])

    if show:
        grid = gridplot([list(item) for item in axes], toolbar_location="above")
        bkp.show(grid)

    return axes
