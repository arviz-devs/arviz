"""Matplotlib Plot posterior densities."""
from typing import Optional
from numbers import Number
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

from . import backend_show
from ....stats import hpd
from ...kdeplot import plot_kde, _fast_kde
from ...plot_utils import (
    make_label,
    _create_axes_grid,
    format_sig_figs,
    round_num,
)


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
    xt_labelsize,
    kwargs,
    titlesize,
    backend_kwargs,
    show,
):
    """Matplotlib posterior plot."""
    if ax is None:
        _, ax = _create_axes_grid(
            length_plotters,
            rows,
            cols,
            figsize=figsize,
            squeeze=False,
            backend_kwargs=backend_kwargs,
        )
    idx = 0
    for (var_name, selection, x), ax_ in zip(plotters, np.ravel(ax)):
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
            xt_labelsize=xt_labelsize,
            **kwargs
        )
        idx += 1
        ax_.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)

    if backend_show(show):
        plt.show()

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
    xt_labelsize,
    round_to: Optional[int] = None,
    **kwargs
):  # noqa: D202
    """Artist to draw posterior."""

    def format_as_percent(x, round_to=0):
        return "{0:.{1:d}f}%".format(100 * x, round_to)

    def display_ref_val():
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
        ax.axvline(val, ymin=0.05, ymax=0.75, color="C1", lw=linewidth, alpha=0.65)
        ax.text(
            values.mean(),
            plot_height * 0.6,
            ref_in_posterior,
            size=ax_labelsize,
            color="C1",
            weight="semibold",
            horizontalalignment="center",
        )

    def display_rope():
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

        ax.plot(
            vals,
            (plot_height * 0.02, plot_height * 0.02),
            lw=linewidth * 5,
            color="C2",
            solid_capstyle="butt",
            zorder=0,
            alpha=0.7,
        )
        text_props = {"size": ax_labelsize, "horizontalalignment": "center", "color": "C2"}
        ax.text(vals[0], plot_height * 0.2, vals[0], weight="semibold", **text_props)
        ax.text(vals[1], plot_height * 0.2, vals[1], weight="semibold", **text_props)

    def display_point_estimate():
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
        ax.text(
            point_value,
            plot_height * 0.8,
            point_text,
            size=ax_labelsize,
            horizontalalignment="center",
        )

    def display_hpd():
        # np.ndarray with 2 entries, min and max
        # pylint: disable=line-too-long
        hpd_intervals = hpd(
            values, credible_interval=credible_interval, multimodal=multimodal
        )  # type: np.ndarray

        for hpdi in np.atleast_2d(hpd_intervals):
            ax.plot(
                hpdi,
                (plot_height * 0.02, plot_height * 0.02),
                lw=linewidth * 2,
                color="k",
                solid_capstyle="butt",
            )
            ax.text(
                hpdi[0],
                plot_height * 0.07,
                round_num(hpdi[0], round_to),
                size=ax_labelsize,
                horizontalalignment="center",
            )
            ax.text(
                hpdi[1],
                plot_height * 0.07,
                round_num(hpdi[1], round_to),
                size=ax_labelsize,
                horizontalalignment="center",
            )
            ax.text(
                (hpdi[0] + hpdi[1]) / 2,
                plot_height * 0.3,
                format_as_percent(credible_interval) + " HPD",
                size=ax_labelsize,
                horizontalalignment="center",
            )

    def format_axes():
        ax.yaxis.set_ticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(
            axis="x", direction="out", width=1, length=3, color="0.5", labelsize=xt_labelsize
        )
        ax.spines["bottom"].set_color("0.5")

    if kind == "kde" and values.dtype.kind == "f":
        kwargs.setdefault("linewidth", linewidth)
        plot_kde(
            values,
            bw=bw,
            fill_kwargs={"alpha": kwargs.pop("fill_alpha", 0)},
            plot_kwargs=kwargs,
            ax=ax,
            rug=False,
        )
    else:
        if bins is None:
            if values.dtype.kind == "i":
                xmin = values.min()
                xmax = values.max()
                bins = range(xmin, xmax + 2)
                ax.set_xlim(xmin - 0.5, xmax + 0.5)
            else:
                bins = "auto"
        kwargs.setdefault("align", "left")
        kwargs.setdefault("color", "C0")
        ax.hist(values, bins=bins, alpha=0.35, **kwargs)

    plot_height = ax.get_ylim()[1]

    format_axes()
    if credible_interval is not None:
        display_hpd()
    display_point_estimate()
    display_ref_val()
    display_rope()
