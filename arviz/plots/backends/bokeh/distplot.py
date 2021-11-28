"""Bokeh Distplot."""
import matplotlib.pyplot as plt
import numpy as np

from ....stats.density_utils import get_bins, histogram
from ...kdeplot import plot_kde
from ...plot_utils import (
    _scale_fig_size,
    set_bokeh_circular_ticks_labels,
    vectorized_to_hex,
    _init_kwargs_dict,
)
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


def plot_dist(
    values,
    values2,
    color,
    kind,
    cumulative,
    label,
    rotated,
    rug,
    bw,
    quantiles,
    contour,
    fill_last,
    figsize,
    textsize,
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
    contour_kwargs,
    contourf_kwargs,
    pcolormesh_kwargs,
    hist_kwargs,
    is_circular,
    ax,
    backend_kwargs,
    show,
):
    """Bokeh distplot."""
    backend_kwargs = _init_kwargs_dict(backend_kwargs)

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, *_ = _scale_fig_size(figsize, textsize)

    color = vectorized_to_hex(color)

    hist_kwargs = _init_kwargs_dict(hist_kwargs)
    if kind == "hist":
        hist_kwargs.setdefault("cumulative", cumulative)
        hist_kwargs.setdefault("fill_color", color)
        hist_kwargs.setdefault("line_color", color)
        hist_kwargs.setdefault("line_alpha", 0)
        if label is not None:
            hist_kwargs.setdefault("legend_label", str(label))

    if ax is None:
        ax = create_axes_grid(
            1,
            figsize=figsize,
            squeeze=True,
            polar=is_circular,
            backend_kwargs=backend_kwargs,
        )

    if kind == "auto":
        kind = "hist" if values.dtype.kind == "i" else "kde"

    if kind == "hist":
        _histplot_bokeh_op(
            values=values,
            values2=values2,
            rotated=rotated,
            ax=ax,
            hist_kwargs=hist_kwargs,
            is_circular=is_circular,
        )
    elif kind == "kde":
        plot_kwargs = _init_kwargs_dict(plot_kwargs)
        if color is None:
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        plot_kwargs.setdefault("line_color", color)
        legend = label is not None

        plot_kde(
            values,
            values2,
            cumulative=cumulative,
            rug=rug,
            label=label,
            bw=bw,
            is_circular=is_circular,
            quantiles=quantiles,
            rotated=rotated,
            contour=contour,
            legend=legend,
            fill_last=fill_last,
            plot_kwargs=plot_kwargs,
            fill_kwargs=fill_kwargs,
            rug_kwargs=rug_kwargs,
            contour_kwargs=contour_kwargs,
            contourf_kwargs=contourf_kwargs,
            pcolormesh_kwargs=pcolormesh_kwargs,
            ax=ax,
            backend="bokeh",
            backend_kwargs={},
            show=False,
        )
    else:
        raise TypeError(f'Invalid "kind":{kind}. Select from {{"auto","kde","hist"}}')

    show_layout(ax, show)

    return ax


def _histplot_bokeh_op(values, values2, rotated, ax, hist_kwargs, is_circular):
    """Add a histogram for the data to the axes."""
    if values2 is not None:
        raise NotImplementedError("Insert hexbin plot here")

    color = hist_kwargs.pop("color", False)
    if color:
        hist_kwargs["fill_color"] = color
        hist_kwargs["line_color"] = color

    bins = hist_kwargs.pop("bins", None)
    if bins is None:
        bins = get_bins(values)
    hist, hist_dens, edges = histogram(np.asarray(values).flatten(), bins=bins)
    if hist_kwargs.pop("density", True):
        hist = hist_dens
    if hist_kwargs.pop("cumulative", False):
        hist = np.cumsum(hist)
        hist /= hist[-1]

    if is_circular:

        if is_circular == "degrees":
            edges = np.deg2rad(edges)
            labels = ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"]
        else:

            labels = [
                r"0",
                r"π/4",
                r"π/2",
                r"3π/4",
                r"π",
                r"5π/4",
                r"3π/2",
                r"7π/4",
            ]

        delta = np.mean(np.diff(edges) / 2)

        ax.annular_wedge(
            x=0,
            y=0,
            inner_radius=0,
            outer_radius=hist,
            start_angle=edges[1:] - delta,
            end_angle=edges[:-1] - delta,
            direction="clock",
            **hist_kwargs,
        )

        ax = set_bokeh_circular_ticks_labels(ax, hist, labels)

    else:

        if rotated:
            ax.quad(top=edges[:-1], bottom=edges[1:], left=0, right=hist, **hist_kwargs)
        else:
            ax.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], **hist_kwargs)

    return ax
