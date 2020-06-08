"""Matplotlib Densityplot."""
import matplotlib.pyplot as plt
import numpy as np

from . import backend_show
from ....stats import hdi
from ...plot_utils import (
    make_label,
    _create_axes_grid,
    calculate_point_estimate,
)
from ....numeric_utils import _fast_kde, get_bins


def plot_density(
    ax,
    all_labels,
    to_plot,
    colors,
    bw,
    figsize,
    length_plotters,
    rows,
    cols,
    titlesize,
    xt_labelsize,
    linewidth,
    markersize,
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
    if ax is None:
        _, ax = _create_axes_grid(
            length_plotters,
            rows,
            cols,
            figsize=figsize,
            squeeze=False,
            backend="matplotlib",
            backend_kwargs=backend_kwargs,
        )
    else:
        ax = np.atleast_2d(ax)

    axis_map = {label: ax_ for label, ax_ in zip(all_labels, np.ravel(ax))}

    for m_idx, plotters in enumerate(to_plot):
        for var_name, selection, values in plotters:
            label = make_label(var_name, selection)
            _d_helper(
                values.flatten(),
                label,
                colors[m_idx],
                bw,
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
            ax.item(0).plot([], label=label, c=colors[m_idx], markersize=markersize)
        ax.item(0).legend(fontsize=xt_labelsize)

    if backend_show(show):
        plt.show()

    return ax


def _d_helper(
    vec,
    vname,
    color,
    bw,
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
    bw : float
        Bandwidth scaling factor. Should be larger than 0. The higher this number the smoother the
        KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule of thumb
        (the default used rule by SciPy).
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

        density, xmin, xmax = _fast_kde(new_vec, bw=bw)
        density *= hdi_prob
        x = np.linspace(xmin, xmax, len(density))
        ymin = density[0]
        ymax = density[-1]

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
