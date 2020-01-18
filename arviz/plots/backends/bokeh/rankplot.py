"""Bokeh rankplot."""
import bokeh.plotting as bkp
import numpy as np
import scipy.stats
from bokeh.layouts import gridplot
from bokeh.models import Span
from bokeh.models.annotations import Title
from bokeh.models.tickers import FixedTicker

from . import backend_kwarg_defaults, backend_show
from ...plot_utils import (
    _create_axes_grid,
    make_label,
)
from ....stats.stats_utils import histogram


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
    """Bokeh rank plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }
    if axes is None:
        _, axes = _create_axes_grid(
            length_plotters,
            rows,
            cols,
            figsize=figsize,
            squeeze=False,
            sharex=True,
            sharey=True,
            backend="bokeh",
            backend_kwargs=backend_kwargs,
        )
    else:
        axes = np.atleast_2d(axes)

    for ax, (var_name, selection, var_data) in zip(
        (item for item in axes.flatten() if item is not None), plotters
    ):
        ranks = scipy.stats.rankdata(var_data).reshape(var_data.shape)
        bin_ary = np.histogram_bin_edges(ranks, bins=bins, range=(0, ranks.size))
        all_counts = np.empty((len(ranks), len(bin_ary) - 1))
        for idx, row in enumerate(ranks):
            _, all_counts[idx], _ = histogram(row, bins=bin_ary)
        counts_normalizer = all_counts.max() / 0.95
        gap = 1
        width = bin_ary[1] - bin_ary[0]

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
                    width=width,
                    fill_color=colors[idx],
                    line_color="white",
                )
                if ref_line:
                    hline = Span(
                        location=y_ticks[-1] + counts.mean(), line_dash="dashed", line_color="black"
                    )
                    ax.add_layout(hline)
            if labels:
                ax.yaxis.axis_label = "Chain"
        elif kind == "vlines":
            ymin = np.full(len(all_counts), all_counts.mean())
            for idx, counts in enumerate(all_counts):
                ax.circle(bin_ary, counts, fill_color=colors[idx], line_color=colors[idx])

                x_locations = [(bin, bin) for bin in bin_ary]
                y_locations = [(ymin[idx], counts_) for counts_ in counts]
                ax.multi_line(
                    x_locations,
                    y_locations,
                    line_dash="solid",
                    line_color=colors[idx],
                    line_width=3,
                )

            if ref_line:
                hline = Span(location=all_counts.mean(), line_dash="dashed", line_color="black")
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
        _title.text = make_label(var_name, selection)
        ax.title = _title

    if backend_show(show):
        grid = gridplot(axes.tolist(), toolbar_location="above")
        bkp.show(grid)

    return axes
