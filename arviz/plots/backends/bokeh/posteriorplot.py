"""Bokeh Plot posterior densities."""

from numbers import Number
from typing import Optional

import numpy as np
from bokeh.models.annotations import Title

from ....stats import hdi
from ....stats.density_utils import get_bins, histogram
from ...kdeplot import plot_kde
from ...plot_utils import (
    _scale_fig_size,
    calculate_point_estimate,
    format_sig_figs,
    round_num,
    vectorized_to_hex,
)
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


def plot_posterior(
    ax,
    length_plotters,
    rows,
    cols,
    figsize,
    plotters,
    bw,
    circular,
    bins,
    kind,
    point_estimate,
    round_to,
    hdi_prob,
    multimodal,
    skipna,
    textsize,
    ref_val,
    rope,
    ref_val_color,
    rope_color,
    labeller,
    kwargs,
    backend_kwargs,
    show,
):
    """Bokeh posterior plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(
            ("dpi", "plot.bokeh.figure.dpi"),
        ),
        **backend_kwargs,
    }

    (figsize, ax_labelsize, *_, linewidth, _) = _scale_fig_size(figsize, textsize, rows, cols)

    if ax is None:
        ax = create_axes_grid(
            length_plotters,
            rows,
            cols,
            figsize=figsize,
            backend_kwargs=backend_kwargs,
        )
    else:
        ax = np.atleast_2d(ax)
    idx = 0
    for (var_name, selection, isel, x), ax_ in zip(
        plotters, (item for item in ax.flatten() if item is not None)
    ):
        _plot_posterior_op(
            idx,
            x.flatten(),
            var_name,
            selection,
            ax=ax_,
            bw=bw,
            circular=circular,
            bins=bins,
            kind=kind,
            point_estimate=point_estimate,
            round_to=round_to,
            hdi_prob=hdi_prob,
            multimodal=multimodal,
            skipna=skipna,
            linewidth=linewidth,
            ref_val=ref_val,
            rope=rope,
            ref_val_color=ref_val_color,
            rope_color=rope_color,
            ax_labelsize=ax_labelsize,
            **kwargs,
        )
        idx += 1
        _title = Title()
        _title.text = labeller.make_label_vert(var_name, selection, isel)
        ax_.title = _title

    show_layout(ax, show)

    return ax


def _plot_posterior_op(
    idx,
    values,
    var_name,
    selection,
    ax,
    bw,
    circular,
    linewidth,
    bins,
    kind,
    point_estimate,
    hdi_prob,
    multimodal,
    skipna,
    ref_val,
    rope,
    ref_val_color,
    rope_color,
    ax_labelsize,
    round_to: Optional[int] = None,
    **kwargs,
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
        ax.line(
            [val, val],
            [0, 0.8 * max_data],
            line_color=vectorized_to_hex(ref_val_color),
            line_alpha=0.65,
        )

        ax.text(
            x=[values.mean()],
            y=[max_data * 0.6],
            text=[ref_in_posterior],
            text_color=vectorized_to_hex(ref_val_color),
            text_align="center",
        )

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
        rope_text = [f"{val:.{format_sig_figs(val, round_to)}g}" for val in vals]

        ax.line(
            vals,
            (max_data * 0.02, max_data * 0.02),
            line_width=linewidth * 5,
            line_color=vectorized_to_hex(rope_color),
            line_alpha=0.7,
        )
        probability_within_rope = ((values > vals[0]) & (values <= vals[1])).mean()
        text_props = dict(
            text_color=vectorized_to_hex(rope_color),
            text_align="center",
        )
        ax.text(
            x=values.mean(),
            y=[max_data * 0.45],
            text=[f"{format_as_percent(probability_within_rope, 1)} in ROPE"],
            **text_props,
        )

        ax.text(
            x=vals,
            y=[max_data * 0.2, max_data * 0.2],
            text_font_size=f"{ax_labelsize}pt",
            text=rope_text,
            **text_props,
        )

    def display_point_estimate(max_data):
        if not point_estimate:
            return
        point_value = calculate_point_estimate(point_estimate, values, bw, circular)
        sig_figs = format_sig_figs(point_value, round_to)
        point_text = "{point_estimate}={point_value:.{sig_figs}g}".format(
            point_estimate=point_estimate, point_value=point_value, sig_figs=sig_figs
        )

        ax.text(x=[point_value], y=[max_data * 0.8], text=[point_text], text_align="center")

    def display_hdi(max_data):
        # np.ndarray with 2 entries, min and max
        # pylint: disable=line-too-long
        hdi_probs = hdi(
            values, hdi_prob=hdi_prob, circular=circular, multimodal=multimodal, skipna=skipna
        )  # type: np.ndarray

        for hdi_i in np.atleast_2d(hdi_probs):
            ax.line(
                hdi_i,
                (max_data * 0.02, max_data * 0.02),
                line_width=linewidth * 2,
                line_color="black",
            )

            ax.text(
                x=list(hdi_i) + [(hdi_i[0] + hdi_i[1]) / 2],
                y=[max_data * 0.07, max_data * 0.07, max_data * 0.3],
                text=(
                    list(map(str, map(lambda x: round_num(x, round_to), hdi_i)))
                    + [f"{format_as_percent(hdi_prob)} HDI"]
                ),
                text_align="center",
            )

    def format_axes():
        ax.yaxis.visible = False
        ax.yaxis.major_tick_line_color = None
        ax.yaxis.minor_tick_line_color = None
        ax.yaxis.major_label_text_font_size = "0pt"
        ax.xgrid.grid_line_color = None
        ax.ygrid.grid_line_color = None

    if skipna:
        values = values[~np.isnan(values)]

    if kind == "kde" and values.dtype.kind == "f":
        kwargs.setdefault("line_width", linewidth)
        plot_kde(
            values,
            bw=bw,
            is_circular=circular,
            fill_kwargs={"fill_alpha": kwargs.pop("fill_alpha", 0)},
            plot_kwargs=kwargs,
            ax=ax,
            rug=False,
            backend="bokeh",
            backend_kwargs={},
            show=False,
        )
        max_data = values.max()
    elif values.dtype.kind == "i" or (values.dtype.kind == "f" and kind == "hist"):
        if bins is None:
            bins = get_bins(values)
        kwargs.setdefault("align", "left")
        kwargs.setdefault("color", "blue")
        _, hist, edges = histogram(values, bins=bins)
        max_data = hist.max()
        ax.quad(
            top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.35, line_alpha=0.35
        )
    elif values.dtype.kind == "b":
        if bins is None:
            bins = "auto"
        kwargs.setdefault("color", "blue")

        hist = np.array([(~values).sum(), values.sum()])
        max_data = hist.max()
        edges = np.array([-0.5, 0.5, 1.5])
        ax.quad(
            top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.35, line_alpha=0.35
        )
        hdi_prob = "hide"
        ax.xaxis.ticker = [0, 1]
        ax.xaxis.major_label_overrides = {0: "False", 1: "True"}
    else:
        raise TypeError("Values must be float, integer or boolean")

    format_axes()
    if hdi_prob != "hide":
        display_hdi(max_data)
    display_point_estimate(max_data)
    display_ref_val(max_data)
    display_rope(max_data)
