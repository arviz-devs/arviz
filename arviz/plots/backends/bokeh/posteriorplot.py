"""Bokeh Plot posterior densities."""
from numbers import Number
from typing import Optional

import bokeh.plotting as bkp
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models.annotations import Title
from scipy.stats import mode

from . import backend_kwarg_defaults, backend_show
from ...kdeplot import plot_kde, _fast_kde
from ...plot_utils import (
    make_label,
    _create_axes_grid,
    format_sig_figs,
    round_num,
)
from ....stats import hpd


def plot_posterior(
    ax,
    length_plotters,
    rows,
    cols,
    figsize,
    plotters,
    bw,
    bins,
    kind,
    point_estimate,
    round_to,
    credible_interval,
    multimodal,
    ref_val,
    rope,
    ax_labelsize,
    kwargs,
    backend_kwargs,
    show,
):
    """Bokeh posterior plot."""
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
    idx = 0
    for (var_name, selection, x), ax_ in zip(
        plotters, (item for item in ax.flatten() if item is not None)
    ):
        _plot_posterior_op(
            idx,
            x.flatten(),
            var_name,
            selection,
            ax=ax_,
            bw=bw,
            bins=bins,
            kind=kind,
            point_estimate=point_estimate,
            round_to=round_to,
            credible_interval=credible_interval,
            multimodal=multimodal,
            ref_val=ref_val,
            rope=rope,
            ax_labelsize=ax_labelsize,
            **kwargs
        )
        idx += 1
        _title = Title()
        _title.text = make_label(var_name, selection)
        ax_.title = _title

    if backend_show(show):
        grid = gridplot(ax.tolist(), toolbar_location="above")
        bkp.show(grid)

    return ax


def _plot_posterior_op(
    idx,
    values,
    var_name,
    selection,
    ax,
    bw,
    linewidth,
    bins,
    kind,
    point_estimate,
    credible_interval,
    multimodal,
    ref_val,
    rope,
    ax_labelsize,
    round_to: Optional[int] = None,
    **kwargs
):  # noqa: D202
    """Artist to draw posterior."""

    def format_as_percent(x, round_to=0):
        return "{0:.{1:d}f}%".format(100 * x, round_to)

    def display_ref_val(max_data):
        if ref_val is None:
            return
        elif isinstance(ref_val, dict):
            val = None
            for sel in ref_val.get(var_name, []):
                if all(
                    k in selection and selection[k] == v for k, v in sel.items() if k != "ref_val"
                ):
                    val = sel["ref_val"]
                    break
            if val is None:
                return
        elif isinstance(ref_val, list):
            val = ref_val[idx]
        elif isinstance(ref_val, Number):
            val = ref_val
        else:
            raise ValueError(
                "Argument `ref_val` must be None, a constant, a list or a "
                'dictionary like {"var_name": [{"ref_val": ref_val}]}'
            )
        less_than_ref_probability = (values < val).mean()
        greater_than_ref_probability = (values >= val).mean()
        ref_in_posterior = "{} <{:g}< {}".format(
            format_as_percent(less_than_ref_probability, 1),
            val,
            format_as_percent(greater_than_ref_probability, 1),
        )
        ax.line([val, val], [0, 0.8 * max_data], line_color="blue", line_alpha=0.65)

        ax.text(x=[values.mean()], y=[max_data * 0.6], text=[ref_in_posterior], text_align="center")

    def display_rope(max_data):
        if rope is None:
            return
        elif isinstance(rope, dict):
            vals = None
            for sel in rope.get(var_name, []):
                # pylint: disable=line-too-long
                if all(k in selection and selection[k] == v for k, v in sel.items() if k != "rope"):
                    vals = sel["rope"]
                    break
            if vals is None:
                return
        elif len(rope) == 2:
            vals = rope
        else:
            raise ValueError(
                "Argument `rope` must be None, a dictionary like"
                '{"var_name": {"rope": (lo, hi)}}, or an'
                "iterable of length 2"
            )

        ax.line(
            vals,
            (max_data * 0.02, max_data * 0.02),
            line_width=linewidth * 5,
            line_color="red",
            line_alpha=0.7,
        )

        text_props = dict(
            text_font_size="{}pt".format(ax_labelsize), text_color="black", text_align="center"
        )

        ax.text(x=vals, y=[max_data * 0.2, max_data * 0.2], text=list(map(str, vals)), **text_props)

    def display_point_estimate(max_data):
        if not point_estimate:
            return
        if point_estimate not in ("mode", "mean", "median"):
            raise ValueError("Point Estimate should be in ('mode','mean','median')")
        if point_estimate == "mean":
            point_value = values.mean()
        elif point_estimate == "mode":
            if isinstance(values[0], float):
                density, lower, upper = _fast_kde(values, bw=bw)
                x = np.linspace(lower, upper, len(density))
                point_value = x[np.argmax(density)]
            else:
                point_value = mode(values)[0][0]
        elif point_estimate == "median":
            point_value = np.median(values)
        sig_figs = format_sig_figs(point_value, round_to)
        point_text = "{point_estimate}={point_value:.{sig_figs}g}".format(
            point_estimate=point_estimate, point_value=point_value, sig_figs=sig_figs
        )

        ax.text(x=[point_value], y=[max_data * 0.8], text=[point_text], text_align="center")

    def display_hpd(max_data):
        # np.ndarray with 2 entries, min and max
        # pylint: disable=line-too-long
        hpd_intervals = hpd(
            values, credible_interval=credible_interval, multimodal=multimodal
        )  # type: np.ndarray

        for hpdi in np.atleast_2d(hpd_intervals):
            ax.line(
                hpdi,
                (max_data * 0.02, max_data * 0.02),
                line_width=linewidth * 2,
                line_color="black",
            )

            ax.text(
                x=list(hpdi) + [(hpdi[0] + hpdi[1]) / 2],
                y=[max_data * 0.07, max_data * 0.07, max_data * 0.3],
                text=list(map(str, map(lambda x: round_num(x, round_to), hpdi)))
                + [format_as_percent(credible_interval) + " HPD"],
                text_align="center",
            )

    def format_axes():
        ax.yaxis.visible = False
        ax.yaxis.major_tick_line_color = None
        ax.yaxis.minor_tick_line_color = None
        ax.yaxis.major_label_text_font_size = "0pt"
        ax.xgrid.grid_line_color = None
        ax.ygrid.grid_line_color = None

    if kind == "kde" and values.dtype.kind == "f":
        kwargs.setdefault("line_width", linewidth)
        plot_kde(
            values,
            bw=bw,
            fill_kwargs={"fill_alpha": kwargs.pop("fill_alpha", 0)},
            plot_kwargs=kwargs,
            ax=ax,
            rug=False,
            backend="bokeh",
            backend_kwargs={},
            show=False,
        )
        hist, edges = np.histogram(values, density=True)
    else:
        if bins is None:
            if values.dtype.kind == "i":
                xmin = values.min()
                xmax = values.max()
                bins = range(xmin, xmax + 2)
            else:
                bins = "auto"
        kwargs.setdefault("align", "left")
        kwargs.setdefault("color", "blue")
        hist, edges = np.histogram(values, density=True, bins=bins)
        ax.quad(
            top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.35, line_alpha=0.35
        )

    format_axes()
    max_data = hist.max()
    if credible_interval is not None:
        display_hpd(max_data)
    display_point_estimate(max_data)
    display_ref_val(max_data)
    display_rope(max_data)
