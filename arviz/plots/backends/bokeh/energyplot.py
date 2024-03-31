"""Bokeh energyplot."""

from itertools import cycle

import numpy as np
from bokeh.models import Label
from bokeh.models.annotations import Legend
from matplotlib.pyplot import rcParams as mpl_rcParams

from ....stats import bfmi as e_bfmi
from ...kdeplot import plot_kde
from ...plot_utils import _scale_fig_size, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
from .distplot import _histplot_bokeh_op


def plot_energy(
    ax,
    energy,
    kind,
    bfmi,
    figsize,
    textsize,
    fill_alpha,
    fill_color,
    fill_kwargs,
    plot_kwargs,
    bw,
    legend,
    backend_kwargs,
    show,
):
    """Bokeh energy plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(("dpi", "plot.bokeh.figure.dpi")),
        **backend_kwargs,
    }
    dpi = backend_kwargs.pop("dpi")

    figsize, _, _, _, line_width, _ = _scale_fig_size(figsize, textsize, 1, 1)

    fill_kwargs = {} if fill_kwargs is None else fill_kwargs
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    plot_kwargs.setdefault("line_width", line_width)
    if kind == "hist":
        legend = False

    if ax is None:
        ax = create_axes_grid(
            1,
            figsize=figsize,
            squeeze=True,
            backend_kwargs=backend_kwargs,
        )

    _colors = [
        prop for _, prop in zip(range(10), cycle(mpl_rcParams["axes.prop_cycle"].by_key()["color"]))
    ]
    if (fill_color[0].startswith("C") and len(fill_color[0]) == 2) and (
        fill_color[1].startswith("C") and len(fill_color[1]) == 2
    ):
        fill_color = tuple((_colors[int(color[1:]) % 10] for color in fill_color))
    elif fill_color[0].startswith("C") and len(fill_color[0]) == 2:
        fill_color = tuple([_colors[int(fill_color[0][1:]) % 10]] + list(fill_color[1:]))
    elif fill_color[1].startswith("C") and len(fill_color[1]) == 2:
        fill_color = tuple(list(fill_color[1:]) + [_colors[int(fill_color[0][1:]) % 10]])

    series = zip(
        fill_alpha,
        fill_color,
        ("Marginal Energy", "Energy transition"),
        (energy - energy.mean(), np.diff(energy)),
    )

    labels = []

    if kind == "kde":
        for alpha, color, label, value in series:
            fill_kwargs["fill_alpha"] = alpha
            fill_kwargs["fill_color"] = vectorized_to_hex(color)
            plot_kwargs["line_alpha"] = alpha
            plot_kwargs["line_color"] = vectorized_to_hex(color)
            _, glyph = plot_kde(
                value,
                bw=bw,
                label=label,
                fill_kwargs=fill_kwargs,
                plot_kwargs=plot_kwargs,
                ax=ax,
                legend=legend,
                backend="bokeh",
                backend_kwargs={},
                show=False,
                return_glyph=True,
            )
            labels.append(
                (
                    label,
                    glyph,
                )
            )

    elif kind == "hist":
        hist_kwargs = plot_kwargs.copy()
        hist_kwargs.update(**fill_kwargs)

        for alpha, color, label, value in series:
            hist_kwargs["fill_alpha"] = alpha
            hist_kwargs["fill_color"] = vectorized_to_hex(color)
            hist_kwargs["line_color"] = None
            hist_kwargs["line_alpha"] = alpha
            _histplot_bokeh_op(
                value.flatten(),
                values2=None,
                rotated=False,
                ax=ax,
                hist_kwargs=hist_kwargs,
                is_circular=False,
            )

    else:
        raise ValueError(f"Plot type {kind} not recognized.")

    if bfmi:
        for idx, val in enumerate(e_bfmi(energy)):
            bfmi_info = Label(
                x=int(figsize[0] * dpi * 0.58),
                y=int(figsize[1] * dpi * 0.73) - 20 * idx,
                x_units="screen",
                y_units="screen",
                text=f"chain {idx:>2} BFMI = {val:.2f}",
                border_line_color=None,
                border_line_alpha=0.0,
                background_fill_color="white",
                background_fill_alpha=1.0,
            )

            ax.add_layout(bfmi_info)

    if legend and label is not None:
        legend = Legend(
            items=labels,
            location="center_right",
            orientation="horizontal",
        )
        ax.add_layout(legend, "above")
        ax.legend.click_policy = "hide"

    show_layout(ax, show)

    return ax
