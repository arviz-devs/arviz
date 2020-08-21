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

    if not color:
        color = "C0"

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
    delta = np.diff(widths).mean()

    for i, width in enumerate(widths):
        tag = False if y[i] == 0 else True
        label = "Positive class" if tag else "Negative class"
        alpha = 0.3 if not tag else 1
        ax.bar(
            width,
            1,
            width=delta,
            color=color,
            align="edge",
            label=label,
            alpha=alpha,
            **plot_kwargs
        )

    if y_hat_line:
        ax.plot(
            np.linspace(0, 1, len(y_hat_var)),
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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if backend_show(show):
        plt.show()

    return ax
