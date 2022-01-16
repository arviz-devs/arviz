"""Bokeh Densityplot."""
from collections import defaultdict
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from bokeh.models.annotations import Legend, Title

from ....stats import hdi
from ....stats.density_utils import get_bins, histogram, kde
from ...plot_utils import _scale_fig_size, calculate_point_estimate, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


def plot_density(
    ax,
    all_labels,
    to_plot,
    colors,
    bw,
    circular,
    figsize,
    length_plotters,
    rows,
    cols,
    textsize,
    labeller,
    hdi_prob,
    point_estimate,
    hdi_markers,
    outline,
    shade,
    n_data,
    data_labels,
    backend_kwargs,
    show,
):
    """Bokeh density plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    if colors == "cycle":
        colors = [
            prop
            for _, prop in zip(
                range(n_data), cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
            )
        ]
    elif isinstance(colors, str):
        colors = [colors for _ in range(n_data)]
    colors = vectorized_to_hex(colors)

    (figsize, _, _, _, line_width, markersize) = _scale_fig_size(figsize, textsize, rows, cols)

    if ax is None:
        ax = create_axes_grid(
            length_plotters,
            rows,
            cols,
            figsize=figsize,
            squeeze=False,
            backend_kwargs=backend_kwargs,
        )
    else:
        ax = np.atleast_2d(ax)

    axis_map = {
        label: ax_
        for label, ax_ in zip(all_labels, (item for item in ax.flatten() if item is not None))
    }
    if data_labels is None:
        data_labels = {}

    legend_items = defaultdict(list)
    for m_idx, plotters in enumerate(to_plot):
        for var_name, selection, isel, values in plotters:
            label = labeller.make_label_vert(var_name, selection, isel)

            if data_labels:
                data_label = data_labels[m_idx]
            else:
                data_label = None

            plotted = _d_helper(
                values.flatten(),
                label,
                colors[m_idx],
                bw,
                circular,
                line_width,
                markersize,
                hdi_prob,
                point_estimate,
                hdi_markers,
                outline,
                shade,
                axis_map[label],
            )
            if data_label is not None:
                legend_items[axis_map[label]].append((data_label, plotted))

    for ax1, legend in legend_items.items():
        legend = Legend(
            items=legend,
            location="center_right",
            orientation="horizontal",
        )
        ax1.add_layout(legend, "above")
        ax1.legend.click_policy = "hide"

    show_layout(ax, show)

    return ax


def _d_helper(
    vec,
    vname,
    color,
    bw,
    circular,
    line_width,
    markersize,
    hdi_prob,
    point_estimate,
    hdi_markers,
    outline,
    shade,
    ax,
):

    extra = {}
    plotted = []

    if vec.dtype.kind == "f":
        if hdi_prob != 1:
            hdi_ = hdi(vec, hdi_prob, multimodal=False)
            new_vec = vec[(vec >= hdi_[0]) & (vec <= hdi_[1])]
        else:
            new_vec = vec

        x, density = kde(new_vec, circular=circular, bw=bw)
        density *= hdi_prob
        xmin, xmax = x[0], x[-1]
        ymin, ymax = density[0], density[-1]

        if outline:
            plotted.append(ax.line(x, density, line_color=color, line_width=line_width, **extra))
            plotted.append(
                ax.line(
                    [xmin, xmin],
                    [-ymin / 100, ymin],
                    line_color=color,
                    line_dash="solid",
                    line_width=line_width,
                    muted_color=color,
                    muted_alpha=0.2,
                )
            )
            plotted.append(
                ax.line(
                    [xmax, xmax],
                    [-ymax / 100, ymax],
                    line_color=color,
                    line_dash="solid",
                    line_width=line_width,
                    muted_color=color,
                    muted_alpha=0.2,
                )
            )

        if shade:
            plotted.append(
                ax.patch(
                    np.r_[x[::-1], x, x[-1:]],
                    np.r_[np.zeros_like(x), density, [0]],
                    fill_color=color,
                    fill_alpha=shade,
                    muted_color=color,
                    muted_alpha=0.2,
                    **extra
                )
            )

    else:
        xmin, xmax = hdi(vec, hdi_prob, multimodal=False)
        bins = get_bins(vec)

        _, hist, edges = histogram(vec, bins=bins)

        if outline:
            plotted.append(
                ax.quad(
                    top=hist,
                    bottom=0,
                    left=edges[:-1],
                    right=edges[1:],
                    line_color=color,
                    fill_color=None,
                    muted_color=color,
                    muted_alpha=0.2,
                    **extra
                )
            )
        else:
            plotted.append(
                ax.quad(
                    top=hist,
                    bottom=0,
                    left=edges[:-1],
                    right=edges[1:],
                    line_color=color,
                    fill_color=color,
                    fill_alpha=shade,
                    muted_color=color,
                    muted_alpha=0.2,
                    **extra
                )
            )

    if hdi_markers:
        plotted.append(ax.diamond(xmin, 0, line_color="black", fill_color=color, size=markersize))
        plotted.append(ax.diamond(xmax, 0, line_color="black", fill_color=color, size=markersize))

    if point_estimate is not None:
        est = calculate_point_estimate(point_estimate, vec, bw, circular)
        plotted.append(ax.circle(est, 0, fill_color=color, line_color="black", size=markersize))

    _title = Title()
    _title.text = vname
    ax.title = _title
    ax.title.text_font_size = "13pt"

    return plotted
