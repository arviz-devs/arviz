"""Matplotlib Densityplot."""
import matplotlib.pyplot as plt
import numpy as np

from . import backend_show
from ....stats import hpd
from ...kdeplot import _fast_kde
from ...plot_utils import _create_axes_grid, make_label


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
    credible_interval,
    point_estimate,
    hpd_markers,
    outline,
    shade,
    n_data,
    data_labels,
    backend_kwargs,
    show,
):
    """Matplotlib densityplot."""
    _, ax = _create_axes_grid(
        length_plotters,
        rows,
        cols,
        figsize=figsize,
        squeeze=False,
        backend="matplotlib",
        backend_kwargs=backend_kwargs,
    )
    axis_map = {label: ax_ for label, ax_ in zip(all_labels, ax.flatten())}

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
                credible_interval,
                point_estimate,
                hpd_markers,
                outline,
                shade,
                axis_map[label],
            )

    if n_data > 1:
        for m_idx, label in enumerate(data_labels):
            ax[0].plot([], label=label, c=colors[m_idx], markersize=markersize)
        ax[0].legend(fontsize=xt_labelsize)

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
    credible_interval,
    point_estimate,
    hpd_markers,
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
    credible_interval : float
        Credible intervals. Defaults to 0.94
    point_estimate : str or None
        'mean' or 'median'
    shade : float
        Alpha blending value for the shaded area under the curve, between 0 (no shade) and 1
        (opaque). Defaults to 0.
    ax : matplotlib axes
    """
    if vec.dtype.kind == "f":
        if credible_interval != 1:
            hpd_ = hpd(vec, credible_interval, multimodal=False)
            new_vec = vec[(vec >= hpd_[0]) & (vec <= hpd_[1])]
        else:
            new_vec = vec

        density, xmin, xmax = _fast_kde(new_vec, bw=bw)
        density *= credible_interval
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
        xmin, xmax = hpd(vec, credible_interval, multimodal=False)
        bins = range(xmin, xmax + 2)
        if outline:
            ax.hist(vec, bins=bins, color=color, histtype="step", align="left")
        if shade:
            ax.hist(vec, bins=bins, color=color, alpha=shade)

    if hpd_markers:
        ax.plot(xmin, 0, hpd_markers, color=color, markeredgecolor="k", markersize=markersize)
        ax.plot(xmax, 0, hpd_markers, color=color, markeredgecolor="k", markersize=markersize)

    if point_estimate is not None:
        if point_estimate == "mean":
            est = np.mean(vec)
        elif point_estimate == "median":
            est = np.median(vec)
        ax.plot(est, 0, "o", color=color, markeredgecolor="k", markersize=markersize)

    ax.set_yticks([])
    ax.set_title(vname, fontsize=titlesize, wrap=True)
    for pos in ["left", "right", "top"]:
        ax.spines[pos].set_visible(False)
    ax.tick_params(labelsize=xt_labelsize)
