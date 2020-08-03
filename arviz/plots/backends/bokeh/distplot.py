"""Bokeh Distplot."""
import bokeh.plotting as bkp
import numpy as np
import matplotlib.pyplot as plt

from . import backend_kwarg_defaults
from .. import show_layout
from ...kdeplot import plot_kde
from ...plot_utils import set_bokeh_circular_ticks_labels
from ....numeric_utils import get_bins


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
    **kwargs  # pylint: disable=unused-argument
):
    """Bokeh distplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    if kind == "hist":
        hist_kwargs.setdefault("cumulative", cumulative)
        hist_kwargs.setdefault("fill_color", color)
        hist_kwargs.setdefault("line_color", color)
        hist_kwargs.setdefault("line_alpha", 0)
        hist_kwargs.setdefault("legend_label", label)

    if ax is None:
        if is_circular:
            ax = bkp.figure(x_axis_type=None, y_axis_type=None)
        else:
            ax = bkp.figure(**backend_kwargs)

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
        if plot_kwargs is None:
            plot_kwargs = {}
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
        raise TypeError('Invalid "kind":{}. Select from {{"auto","kde","hist"}}'.format(kind))

    show_layout(ax, show)

    return ax


def _histplot_bokeh_op(values, values2, rotated, ax, hist_kwargs, is_circular):
    """Add a histogram for the data to the axes."""
    if values2 is not None:
        raise NotImplementedError("Insert hexbin plot here")

    legend_label = hist_kwargs.pop("label", None)
    if legend_label:
        hist_kwargs["legend_label"] = legend_label

    color = hist_kwargs.pop("color", False)
    if color:
        hist_kwargs["fill_color"] = color
        hist_kwargs["line_color"] = color

    bins = hist_kwargs.pop("bins", None)
    if bins is None:
        bins = get_bins(values)
    density = hist_kwargs.pop("density", True)
    hist, edges = np.histogram(np.asarray(values).flatten(), density=density, bins=bins)
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
