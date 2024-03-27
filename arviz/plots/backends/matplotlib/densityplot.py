"""Matplotlib Densityplot."""

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from ....stats import hdi
from ....stats.density_utils import get_bins, kde
from ...plot_utils import _scale_fig_size, calculate_point_estimate
from . import backend_kwarg_defaults, backend_show, create_axes_grid


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
    """Matplotlib densityplot."""
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

    (figsize, _, titlesize, xt_labelsize, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs.setdefault("squeeze", False)
    if ax is None:
        _, ax = create_axes_grid(
            length_plotters,
            rows,
            cols,
            backend_kwargs=backend_kwargs,
        )

    axis_map = dict(zip(all_labels, np.ravel(ax)))

    for m_idx, plotters in enumerate(to_plot):
        for var_name, selection, isel, values in plotters:
            label = labeller.make_label_vert(var_name, selection, isel)
            _d_helper(
                values.flatten(),
                label,
                colors[m_idx],
                bw,
                circular,
                titlesize,
                xt_labelsize,
                linewidth,
                markersize,
                hdi_prob,
                point_estimate,
                hdi_markers,
                outline,
                shade,
                axis_map[label],
            )

    if n_data > 1:
        for m_idx, label in enumerate(data_labels):
            np.ravel(ax).item(0).plot([], label=label, c=colors[m_idx], markersize=markersize)
        np.ravel(ax).item(0).legend(fontsize=xt_labelsize)

    if backend_show(show):
        plt.show()

    return ax


def _d_helper(
    vec,
    vname,
    color,
    bw,
    circular,
    titlesize,
    xt_labelsize,
    linewidth,
    markersize,
    hdi_prob,
    point_estimate,
    hdi_markers,
    outline,
    shade,
    ax,
):
    """Plot an individual dimension.

    Parameters
    ----------
    vec : array
        1D array from trace
    vname : str
        variable name
    color : str
        matplotlib color
    bw: float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "isj" or "experimental" when `circular` is False
        and "taylor" (for now) when `circular` is True.
    titlesize : float
        font size for title
    xt_labelsize : float
       fontsize for xticks
    linewidth : float
        Thickness of lines
    markersize : float
        Size of markers
    hdi_prob : float
        Probability for the highest density interval. Defaults to 0.94
    point_estimate : Optional[str]
        Plot point estimate per variable. Values should be 'mean', 'median', 'mode' or None.
        Defaults to 'auto' i.e. it falls back to default set in rcParams.
    shade : float
        Alpha blending value for the shaded area under the curve, between 0 (no shade) and 1
        (opaque). Defaults to 0.
    ax : matplotlib axes
    """
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
            ax.plot(x, density, color=color, lw=linewidth)
            ax.plot([xmin, xmin], [-ymin / 100, ymin], color=color, ls="-", lw=linewidth)
            ax.plot([xmax, xmax], [-ymax / 100, ymax], color=color, ls="-", lw=linewidth)

        if shade:
            ax.fill_between(x, density, color=color, alpha=shade)

    else:
        xmin, xmax = hdi(vec, hdi_prob, multimodal=False)
        bins = get_bins(vec)
        if outline:
            ax.hist(vec, bins=bins, color=color, histtype="step", align="left")
        if shade:
            ax.hist(vec, bins=bins, color=color, alpha=shade)

    if hdi_markers:
        ax.plot(xmin, 0, hdi_markers, color=color, markeredgecolor="k", markersize=markersize)
        ax.plot(xmax, 0, hdi_markers, color=color, markeredgecolor="k", markersize=markersize)

    if point_estimate is not None:
        est = calculate_point_estimate(point_estimate, vec, bw)
        ax.plot(est, 0, "o", color=color, markeredgecolor="k", markersize=markersize)

    ax.set_yticks([])
    ax.set_title(vname, fontsize=titlesize, wrap=True)
    for pos in ["left", "right", "top"]:
        ax.spines[pos].set_visible(False)
    ax.tick_params(labelsize=xt_labelsize)
