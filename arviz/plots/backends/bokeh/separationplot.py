"""Bokeh separation plot."""
import numpy as np

from ...plot_utils import _scale_fig_size, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


def plot_separation(
    y,
    y_hat,
    y_hat_line,
    label_y_hat,
    expected_events,
    figsize,
    textsize,
    color,
    legend,
    locs,
    width,
    ax,
    plot_kwargs,
    y_hat_line_kwargs,
    exp_events_kwargs,
    backend_kwargs,
    show,
):
    """Matplotlib separation plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    if plot_kwargs is None:
        plot_kwargs = {}

    # plot_kwargs.setdefault("color", "#2a2eec")
    # if color:
    plot_kwargs["color"] = vectorized_to_hex(color)

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    if y_hat_line_kwargs is None:
        y_hat_line_kwargs = {}

    y_hat_line_kwargs.setdefault("color", "black")
    y_hat_line_kwargs.setdefault("line_width", 2)

    if exp_events_kwargs is None:
        exp_events_kwargs = {}

    exp_events_kwargs.setdefault("color", "black")
    exp_events_kwargs.setdefault("size", 15)

    if legend:
        y_hat_line_kwargs.setdefault("legend_label", label_y_hat)
        exp_events_kwargs.setdefault(
            "legend_label",
            "Expected events",
        )

    figsize, *_ = _scale_fig_size(figsize, textsize)

    idx = np.argsort(y_hat)

    backend_kwargs["x_range"] = (0, 1)
    backend_kwargs["y_range"] = (0, 1)

    if ax is None:
        ax = create_axes_grid(1, figsize=figsize, squeeze=True, backend_kwargs=backend_kwargs)

    for i, loc in enumerate(locs):
        positive = not y[idx][i] == 0
        alpha = 1 if positive else 0.3
        ax.vbar(
            loc,
            top=1,
            width=width,
            fill_alpha=alpha,
            line_alpha=alpha,
            **plot_kwargs,
        )

    if y_hat_line:
        ax.line(
            np.linspace(0, 1, len(y_hat)),
            y_hat[idx],
            **y_hat_line_kwargs,
        )

    if expected_events:
        expected_events = int(np.round(np.sum(y_hat)))
        ax.triangle(
            y_hat[idx][len(y_hat) - expected_events - 1],
            0,
            **exp_events_kwargs,
        )

    ax.axis.visible = False
    ax.xgrid.grid_line_color = None
    ax.ygrid.grid_line_color = None

    show_layout(ax, show)

    return ax
