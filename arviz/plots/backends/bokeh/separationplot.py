"""Matplotlib separation plot"""
import numpy as np

from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, create_axes_grid
from .. import show_layout


def plot_separation(
    idata,
    y,
    y_hat,
    y_hat_line,
    expected_events,
    figsize,
    textsize,
    color,
    legend,  # pylint: disable=unused-argument
    ax,
    plot_kwargs,
    backend_kwargs,
    show,
):
    """Matplotlib separation plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    if plot_kwargs is None:
        plot_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    if not color:
        color = "blue"

    figsize, *_ = _scale_fig_size(figsize, textsize)
    if isinstance(y_hat, str):
        y_hat_var = idata.posterior_predictive[y_hat].values.mean(1).mean(0)
        label_line = y_hat

    idx = np.argsort(y_hat_var)

    if isinstance(y, str):
        y = idata.observed_data[y].values[idx].ravel()

    widths = np.linspace(0, 1, len(y_hat_var))
    delta = np.diff(widths).mean()

    backend_kwargs["x_range"] = (0, 1)
    backend_kwargs["y_range"] = (0, 1)

    if ax is None:
        ax = create_axes_grid(1, figsize=figsize, squeeze=True, backend_kwargs=backend_kwargs,)

    for i, width in enumerate(widths):
        tag = False if y[i] == 0 else True
        label = "Positive class" if tag else "Negative class"
        alpha = 0.3 if not tag else 1
        ax.vbar(
            width,
            top=1,
            width=delta,
            color=color,
            fill_alpha=alpha,
            legend_label=label,
            **plot_kwargs
        )

    if y_hat_line:
        ax.line(
            np.linspace(0, 1, len(y_hat_var)),
            y_hat_var[idx],
            color="black",
            legend_label=label_line,
        )

    if expected_events:
        expected_events = int(np.round(np.sum(y_hat_var)))
        ax.triangle(
            y_hat_var[idx][expected_events],
            0,
            color="black",
            legend_label="Expected events",
            **plot_kwargs
        )

    ax.axis.visible = False
    ax.xgrid.grid_line_color = None
    ax.ygrid.grid_line_color = None

    show_layout(ax, show)

    return ax
