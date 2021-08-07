"""Matplotlib plot time series figure."""
import matplotlib.pyplot as plt
import numpy as np

from ...plot_utils import _scale_fig_size
from . import create_axes_grid, backend_show, backend_kwarg_defaults


def plot_ts(
    x_plotters,
    y_plotters,
    y_mean_plotters,
    y_hat_plotters,
    y_holdout_plotters,
    x_holdout_plotters,
    y_forecasts_plotters,
    y_forecasts_mean_plotters,
    num_samples,
    backend_kwargs,
    rows,
    cols,
    textsize,
    figsize,
    axes,
    show,
):
    """Matplotlib time series."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    if figsize is None:
        figsize = (12, rows * 5)

    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs.setdefault("squeeze", False)
    backend_kwargs.setdefault("sharex", True)

    figsize, _, _, _, _, _ = _scale_fig_size(figsize, textsize, rows, cols)

    if axes is None:
        _, axes = create_axes_grid(rows, rows, cols, backend_kwargs=backend_kwargs)

    for i, ax_i in enumerate(np.ravel(axes)[:rows]):
        _, _, _, y_plotters_i = y_plotters[i]
        _, _, _, x_plotters_i = x_plotters[i]

        ax_i.plot(x_plotters_i, y_plotters_i, color="blue")

        if y_hat_plotters is not None:
            *_, y_hat_plotters_i = y_hat_plotters[i]
            *_, x_hat_plotters_i = x_plotters[i]
            for j in range(num_samples):
                ax_i.plot(
                    x_hat_plotters_i,
                    y_hat_plotters_i[..., j],
                    color="grey",
                    alpha=0.1,
                )

            *_, x_mean_plotters_i = x_plotters[i]
            *_, y_mean_plotters_i = y_mean_plotters[i]
            ax_i.plot(x_mean_plotters_i, y_mean_plotters_i, color="red", linestyle="dashed")

        if y_holdout_plotters is not None:
            *_, y_holdout_plotters_i = y_holdout_plotters[i]
            *_, x_holdout_plotters_i = x_holdout_plotters[i]

            ax_i.plot(x_holdout_plotters_i, y_holdout_plotters_i, color="blue")
            ax_i.axvline(x_plotters_i[-1], linestyle="dashed", color="black")

        if y_forecasts_plotters is not None:
            *_, y_forecasts_plotters_i = y_forecasts_plotters[i]
            *_, x_forecasts_plotters_i = x_holdout_plotters[i]
            for j in range(num_samples):
                ax_i.plot(
                    x_forecasts_plotters_i,
                    y_forecasts_plotters_i[..., j],
                    color="grey",
                    alpha=0.1,
                )

            *_, x_forecasts_mean_plotters_i = x_holdout_plotters[i]
            *_, y_forecasts_mean_plotters_i = y_forecasts_mean_plotters[i]
            ax_i.plot(
                x_forecasts_mean_plotters_i,
                y_forecasts_mean_plotters_i,
                color="red",
                linestyle="dashed",
            )
            ax_i.axvline(x_plotters_i[-1], linestyle="dashed", color="black")

    if backend_show(show):
        plt.show()

    return axes
