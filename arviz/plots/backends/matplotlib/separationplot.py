"""Matplotlib separation plot."""
import warnings
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ....data import InferenceData
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

    plot_kwargs.setdefault("color", "C0")
    if color:
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

    if idata is not None and not isinstance(idata, InferenceData):
        raise ValueError("idata must be of type InferenceData or None")

    if idata is None:
        if not all(isinstance(arg, (np.ndarray, xr.DataArray)) for arg in (y, y_hat)):
            raise ValueError(
                "y and y_hat must be array or DataArray when idata is None "
                "but they are of types {}".format([type(arg) for arg in (y, y_hat)])
            )
    else:

        if y_hat is None and isinstance(y, str):
            label_y_hat = y
            y_hat = y
        elif y_hat is None:
            raise ValueError("y_hat cannot be None if y is not a str")

        if isinstance(y, str):
            y = idata.observed_data[y].values
        elif not isinstance(y, (np.ndarray, xr.DataArray)):
            raise ValueError("y must be of types array, DataArray or str, not {}".format(type(y)))

        if isinstance(y_hat, str):
            label_y_hat = y_hat
            y_hat = idata.posterior_predictive[y_hat].mean(axis=(1, 0)).values
        elif not isinstance(y_hat, (np.ndarray, xr.DataArray)):
            raise ValueError(
                "y_hat must be of types array, DataArray or str, not {}".format(type(y_hat))
            )

    idx = np.argsort(y_hat)

    if len(y) != len(y_hat):
        warnings.warn(
            "y and y_hat must be the same lenght", UserWarning,
        )

    locs = np.linspace(0, 1, len(y_hat))
    width = np.diff(locs).mean()

    for i, loc in enumerate(locs):
        positive = not y[idx][i] == 0
        alpha = 1 if positive else 0.3
        ax.bar(loc, 1, width=width, alpha=alpha, **plot_kwargs)

    if y_hat_line:
        ax.plot(np.linspace(0, 1, len(y_hat)), y_hat[idx], label=label_y_hat, **y_hat_line_kwargs)

    if expected_events:
        expected_events = int(np.round(np.sum(y_hat)))
        ax.scatter(y_hat[idx][expected_events - 1], 0, label="Expected events", **exp_events_kwargs)

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
