"""Bokeh Densityplot."""
import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models.annotations import Title

from . import backend_kwarg_defaults, backend_show
from ...kdeplot import _fast_kde
from ...plot_utils import _create_axes_grid, make_label
from ....stats import hpd
from ....stats.stats_utils import histogram


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
    line_width,
    markersize,
    credible_interval,
    point_estimate,
    hpd_markers,
    outline,
    shade,
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

    if ax is None:
        _, ax = _create_axes_grid(
            length_plotters,
            rows,
            cols,
            figsize=figsize,
            squeeze=False,
            backend="bokeh",
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

    for m_idx, plotters in enumerate(to_plot):
        for ax_idx, (var_name, selection, values) in enumerate(plotters):
            label = make_label(var_name, selection)

            if data_labels:
                data_label = data_labels[m_idx]
                if ax_idx != 0 or data_label == "":
                    data_label = None
            else:
                data_label = None

            _d_helper(
                values.flatten(),
                label,
                colors[m_idx],
                bw,
                line_width,
                markersize,
                credible_interval,
                point_estimate,
                hpd_markers,
                outline,
                shade,
                axis_map[label],
                data_label=data_label,
            )

    if backend_show(show):
        grid = gridplot(ax.tolist(), toolbar_location="above")
        bkp.show(grid)

    return ax


def _d_helper(
    vec,
    vname,
    color,
    bw,
    line_width,
    markersize,
    credible_interval,
    point_estimate,
    hpd_markers,
    outline,
    shade,
    ax,
    data_label,
):
    extra = dict()
    if data_label is not None:
        extra["legend_label"] = data_label

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
            ax.line(x, density, line_color=color, line_width=line_width, **extra)
            ax.line(
                [xmin, xmin],
                [-ymin / 100, ymin],
                line_color=color,
                line_dash="solid",
                line_width=line_width,
            )
            ax.line(
                [xmax, xmax],
                [-ymax / 100, ymax],
                line_color=color,
                line_dash="solid",
                line_width=line_width,
            )

        if shade:
            ax.patch(
                np.r_[x[::-1], x, x[-1:]],
                np.r_[np.zeros_like(x), density, [0]],
                fill_color=color,
                fill_alpha=shade,
                **extra
            )

    else:
        xmin, xmax = hpd(vec, credible_interval, multimodal=False)
        bins = range(xmin, xmax + 2)

        _, hist, edges = histogram(vec, bins=bins)

        if outline:
            ax.quad(
                top=hist,
                bottom=0,
                left=edges[:-1],
                right=edges[1:],
                line_color=color,
                fill_color=None,
                **extra
            )
        else:
            ax.quad(
                top=hist,
                bottom=0,
                left=edges[:-1],
                right=edges[1:],
                line_color=color,
                fill_color=color,
                fill_alpha=shade,
                **extra
            )

    if hpd_markers:
        ax.diamond(xmin, 0, line_color="black", fill_color=color, size=markersize)
        ax.diamond(xmax, 0, line_color="black", fill_color=color, size=markersize)

    if point_estimate is not None:
        if point_estimate == "mean":
            est = np.mean(vec)
        elif point_estimate == "median":
            est = np.median(vec)
        ax.circle(est, 0, fill_color=color, line_color="black", size=markersize)

    _title = Title()
    _title.text = vname
    ax.title = _title
