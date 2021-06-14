"""Matplotlib Plot posterior densities."""
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np

from ....stats import hdi
from ....stats.density_utils import get_bins
from ...kdeplot import plot_kde
from ...plot_utils import (
    _scale_fig_size,
    calculate_point_estimate,
    format_sig_figs,
    round_num,
    vectorized_to_hex,
)
from . import backend_kwarg_defaults, backend_show, create_axes_grid, matplotlib_kwarg_dealiaser


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
    """Matplotlib posterior plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    (figsize, ax_labelsize, titlesize, xt_labelsize, _linewidth, _) = _scale_fig_size(
        figsize, textsize, rows, cols
    )
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs.setdefault("squeeze", True)

    if kind == "hist":
        kwargs = matplotlib_kwarg_dealiaser(kwargs, "hist")
    else:
        kwargs = matplotlib_kwarg_dealiaser(kwargs, "plot")
    kwargs.setdefault("linewidth", _linewidth)

    if ax is None:
        _, ax = create_axes_grid(
            length_plotters,
            rows,
            cols,
            backend_kwargs=backend_kwargs,
        )
    idx = 0
    for (var_name, selection, isel, x), ax_ in zip(plotters, np.ravel(ax)):
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
            ref_val=ref_val,
            rope=rope,
            ref_val_color=ref_val_color,
            rope_color=rope_color,
            ax_labelsize=ax_labelsize,
            xt_labelsize=xt_labelsize,
            **kwargs,
        )
        idx += 1
        ax_.set_title(
            labeller.make_label_vert(var_name, selection, isel), fontsize=titlesize, wrap=True
        )

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
    xt_labelsize,
    round_to=None,
    **kwargs,
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
        ax.axvline(
            val,
            ymin=0.05,
            ymax=0.75,
            lw=linewidth,
            alpha=0.65,
            color=vectorized_to_hex(ref_val_color),
        )
        ax.text(
            values.mean(),
            plot_height * 0.6,
            ref_in_posterior,
            size=ax_labelsize,
            weight="semibold",
            horizontalalignment="center",
            color=vectorized_to_hex(ref_val_color),
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
        rope_text = [f"{val:.{format_sig_figs(val, round_to)}g}" for val in vals]
        ax.plot(
            vals,
            (plot_height * 0.02, plot_height * 0.02),
            lw=linewidth * 5,
            solid_capstyle="butt",
            zorder=0,
            alpha=0.7,
            color=vectorized_to_hex(rope_color),
        )
        probability_within_rope = ((values > vals[0]) & (values <= vals[1])).mean()
        ax.text(
            values.mean(),
            plot_height * 0.45,
            f"{format_as_percent(probability_within_rope, 1)} in ROPE",
            weight="semibold",
            horizontalalignment="center",
            size=ax_labelsize,
            color=vectorized_to_hex(rope_color),
        )
        ax.text(
            vals[0],
            plot_height * 0.2,
            rope_text[0],
            weight="semibold",
            horizontalalignment="right",
            size=ax_labelsize,
            color=vectorized_to_hex(rope_color),
        )
        ax.text(
            vals[1],
            plot_height * 0.2,
            rope_text[1],
            weight="semibold",
            horizontalalignment="left",
            size=ax_labelsize,
            color=vectorized_to_hex(rope_color),
        )

    def display_point_estimate():
        if not point_estimate:
            return
        point_value = calculate_point_estimate(point_estimate, values, bw, circular, skipna)
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

    def display_hdi():
        # np.ndarray with 2 entries, min and max
        # pylint: disable=line-too-long
        hdi_probs = hdi(
            values, hdi_prob=hdi_prob, circular=circular, multimodal=multimodal, skipna=skipna
        )  # type: np.ndarray

        for hdi_i in np.atleast_2d(hdi_probs):
            ax.plot(
                hdi_i,
                (plot_height * 0.02, plot_height * 0.02),
                lw=linewidth * 2,
                color="k",
                solid_capstyle="butt",
            )
            ax.text(
                hdi_i[0],
                plot_height * 0.07,
                round_num(hdi_i[0], round_to) + " ",
                size=ax_labelsize,
                horizontalalignment="right",
            )
            ax.text(
                hdi_i[1],
                plot_height * 0.07,
                " " + round_num(hdi_i[1], round_to),
                size=ax_labelsize,
                horizontalalignment="left",
            )
            ax.text(
                (hdi_i[0] + hdi_i[1]) / 2,
                plot_height * 0.3,
                format_as_percent(hdi_prob) + " HDI",
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
            is_circular=circular,
            fill_kwargs={"alpha": kwargs.pop("fill_alpha", 0)},
            plot_kwargs=kwargs,
            ax=ax,
            rug=False,
            show=False,
        )
    else:
        if bins is None:
            if values.dtype.kind == "i":
                xmin = values.min()
                xmax = values.max()
                bins = get_bins(values)
                ax.set_xlim(xmin - 0.5, xmax + 0.5)
            else:
                bins = "auto"
        kwargs.setdefault("align", "left")
        kwargs.setdefault("color", "C0")
        ax.hist(values, bins=bins, alpha=0.35, **kwargs)

    plot_height = ax.get_ylim()[1]

    format_axes()
    if hdi_prob != "hide":
        display_hdi()
    display_point_estimate()
    display_ref_val()
    display_rope()
