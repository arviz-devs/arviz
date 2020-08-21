"""Matplotlib separation plot"""
import matplotlib.pyplot as plt
import numpy as np

from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show, create_axes_grid


def plot_separation(
    idata,
    y,
    y_hat,
    y_hat_line,
    expected_events,
    figsize,
    textsize,
    color,
    cmap,
    legend,
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

    if cmap:
        cmap = plt.get_cmap(cmap).colors
        negative_color, positive_color = cmap[-1], cmap[0]
    else:
        if color:
            negative_color, positive_color = color[0], color[1]
        else:
            negative_color, positive_color = "C1", "C3"

    (figsize, *_) = _scale_fig_size(figsize, textsize, 1, 1)
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True

    if ax is None:
        _, ax = create_axes_grid(1, backend_kwargs=backend_kwargs)

    if isinstance(y_hat, str):
        y_hat_var = idata.posterior_predictive[y_hat].values.mean(1).mean(0)
        label_line = y_hat

    idx = np.argsort(y_hat_var)

    if isinstance(y, str):
        y = idata.observed_data[y].values[idx].ravel()

    widths = np.linspace(0, 1, len(y_hat_var))

    for i, width in enumerate(widths):
        bar_color, tag = (negative_color, False) if y[i] == 0 else (positive_color, True)
        label = "Positive class" if tag else "Negative class"
        ax.bar(width, 1, width=width, color=bar_color, align="edge", label=label, **plot_kwargs)

    delta = np.diff(widths).mean()

    if y_hat_line:
        ax.plot(
            np.linspace(delta, 1.5, len(y_hat_var)),
            y_hat_var[idx],
            color="k",
            label=label_line,
            **plot_kwargs
        )

    if expected_events:
        expected_events = int(np.round(np.sum(y_hat_var)))
        ax.scatter(
            y_hat_var[idx][expected_events], 0, marker="^", color="k", label="Expected events",
        )

    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        labels_dict = dict(zip(labels, handles))
        ax.legend(labels_dict.values(), labels_dict.keys())

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(delta, 1.5)
    ax.set_ylim(0, 1)

    if backend_show(show):
        plt.show()

    return ax
