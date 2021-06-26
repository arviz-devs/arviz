"""Matplotlib separation plot."""
import matplotlib.pyplot as plt
import numpy as np

from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show, create_axes_grid


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

    # plot_kwargs.setdefault("color", "C0")
    # if color:
    plot_kwargs["color"] = color

    if y_hat_line_kwargs is None:
        y_hat_line_kwargs = {}

    y_hat_line_kwargs.setdefault("color", "k")

    if exp_events_kwargs is None:
        exp_events_kwargs = {}

    exp_events_kwargs.setdefault("color", "k")
    exp_events_kwargs.setdefault("marker", "^")
    exp_events_kwargs.setdefault("s", 100)
    exp_events_kwargs.setdefault("zorder", 2)

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    (figsize, *_) = _scale_fig_size(figsize, textsize, 1, 1)
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True

    if ax is None:
        _, ax = create_axes_grid(1, backend_kwargs=backend_kwargs)

    idx = np.argsort(y_hat)

    for i, loc in enumerate(locs):
        positive = not y[idx][i] == 0
        alpha = 1 if positive else 0.3
        ax.bar(loc, 1, width=width, alpha=alpha, **plot_kwargs)

    if y_hat_line:
        ax.plot(np.linspace(0, 1, len(y_hat)), y_hat[idx], label=label_y_hat, **y_hat_line_kwargs)

    if expected_events:
        expected_events = int(np.round(np.sum(y_hat)))
        ax.scatter(
            y_hat[idx][len(y_hat) - expected_events - 1],
            0,
            label="Expected events",
            **exp_events_kwargs
        )

    if legend and (expected_events or y_hat_line):
        handles, labels = ax.get_legend_handles_labels()
        labels_dict = dict(zip(labels, handles))
        ax.legend(labels_dict.values(), labels_dict.keys())

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if backend_show(show):
        plt.show()

    return ax
