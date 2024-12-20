"""Bokeh rankplot."""

import numpy as np

from bokeh.models import Span
from bokeh.models.annotations import Title
from bokeh.models.tickers import FixedTicker

from ....stats.density_utils import histogram
from ...plot_utils import _scale_fig_size, compute_ranks
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


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
    """Bokeh rank plot."""
    if ref_line_kwargs is None:
        ref_line_kwargs = {}
    ref_line_kwargs.setdefault("line_dash", "dashed")
    ref_line_kwargs.setdefault("line_color", "black")

    if bar_kwargs is None:
        bar_kwargs = {}
    bar_kwargs.setdefault("line_color", "white")

    if vlines_kwargs is None:
        vlines_kwargs = {}
    vlines_kwargs.setdefault("line_width", 2)
    vlines_kwargs.setdefault("line_dash", "solid")

    if marker_vlines_kwargs is None:
        marker_vlines_kwargs = {}
    marker_vlines_kwargs.setdefault("marker", "circle")

    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(
            ("dpi", "plot.bokeh.figure.dpi"),
        ),
        **backend_kwargs,
    }
    figsize, *_ = _scale_fig_size(figsize, None, rows=rows, cols=cols)
    if axes is None:
        axes = create_axes_grid(
            length_plotters,
            rows,
            cols,
            figsize=figsize,
            sharex=True,
            sharey=True,
            backend_kwargs=backend_kwargs,
        )
    else:
        axes = np.atleast_2d(axes)

    for ax, (var_name, selection, isel, var_data) in zip(
        (item for item in axes.flatten() if item is not None), plotters
    ):
        ranks = compute_ranks(var_data)
        bin_ary = np.histogram_bin_edges(ranks, bins=bins, range=(0, ranks.size))
        all_counts = np.empty((len(ranks), len(bin_ary) - 1))
        for idx, row in enumerate(ranks):
            _, all_counts[idx], _ = histogram(row, bins=bin_ary)
        counts_normalizer = all_counts.max() / 0.95
        gap = 1
        width = bin_ary[1] - bin_ary[0]

        bar_kwargs.setdefault("width", width)
        # Center the bins
        bin_ary = (bin_ary[1:] + bin_ary[:-1]) / 2

        y_ticks = []
        if kind == "bars":
            for idx, counts in enumerate(all_counts):
                counts = counts / counts_normalizer
                y_ticks.append(idx * gap)
                ax.vbar(
                    x=bin_ary,
                    top=y_ticks[-1] + counts,
                    bottom=y_ticks[-1],
                    fill_color=colors[idx],
                    **bar_kwargs,
                )
                if ref_line:
                    hline = Span(location=y_ticks[-1] + counts.mean(), **ref_line_kwargs)
                    ax.add_layout(hline)
            if labels:
                ax.yaxis.axis_label = "Chain"
        elif kind == "vlines":
            ymin = np.full(len(all_counts), all_counts.mean())
            for idx, counts in enumerate(all_counts):
                ax.scatter(
                    bin_ary,
                    counts,
                    fill_color=colors[idx],
                    line_color=colors[idx],
                    **marker_vlines_kwargs,
                )
                x_locations = [(bin, bin) for bin in bin_ary]
                y_locations = [(ymin[idx], counts_) for counts_ in counts]
                ax.multi_line(x_locations, y_locations, line_color=colors[idx], **vlines_kwargs)

            if ref_line:
                hline = Span(location=all_counts.mean(), **ref_line_kwargs)
                ax.add_layout(hline)

        if labels:
            ax.xaxis.axis_label = "Rank (all chains)"

            ax.yaxis.ticker = FixedTicker(ticks=y_ticks)
            ax.xaxis.major_label_overrides = dict(
                zip(map(str, y_ticks), map(str, range(len(y_ticks))))
            )

        else:
            ax.yaxis.major_tick_line_color = None
            ax.yaxis.minor_tick_line_color = None

            ax.xaxis.major_label_text_font_size = "0pt"
            ax.yaxis.major_label_text_font_size = "0pt"

        _title = Title()
        _title.text = labeller.make_label_vert(var_name, selection, isel)
        ax.title = _title

    show_layout(axes, show)

    return axes
