"""Bokeh separation plot."""
import warnings
import numpy as np
import xarray as xr

from ....data import InferenceData
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


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

    plot_kwargs.setdefault("color", "#2a2eec")
    if color:
        plot_kwargs["color"] = color

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

    figsize, *_ = _scale_fig_size(figsize, textsize)

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
            "y and y_hat must be the same lenght",
            UserWarning,
        )

    locs = np.linspace(0, 1, len(y_hat))
    width = np.diff(locs).mean()

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
            legend_label=label_y_hat,
            **y_hat_line_kwargs,
        )

    if expected_events:
        expected_events = int(np.round(np.sum(y_hat)))
        ax.triangle(
            y_hat[idx][expected_events - 1],
            0,
            legend_label="Expected events",
            **exp_events_kwargs,
        )

    ax.axis.visible = False
    ax.xgrid.grid_line_color = None
    ax.ygrid.grid_line_color = None

    show_layout(ax, show)

    return ax
