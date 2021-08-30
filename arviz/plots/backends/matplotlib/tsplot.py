"""Matplotlib plot time series figure."""
import matplotlib.pyplot as plt
import numpy as np

from ...plot_utils import _scale_fig_size
from . import create_axes_grid, backend_show, matplotlib_kwarg_dealiaser, backend_kwarg_defaults


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
    y_kwargs,
    y_hat_plot_kwargs,
    y_mean_plot_kwargs,
    vline_kwargs,
    length_plotters,
    rows,
    cols,
    textsize,
    figsize,
    legend,
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

    figsize, _, _, xt_labelsize, _, _ = _scale_fig_size(figsize, textsize, rows, cols)

    if axes is None:
        _, axes = create_axes_grid(length_plotters, rows, cols, backend_kwargs=backend_kwargs)

    y_kwargs = matplotlib_kwarg_dealiaser(y_kwargs, "plot")
    y_kwargs.setdefault("color", "blue")
    y_kwargs.setdefault("zorder", 10)

    y_hat_plot_kwargs = matplotlib_kwarg_dealiaser(y_hat_plot_kwargs, "plot")
    y_hat_plot_kwargs.setdefault("color", "grey")
    y_hat_plot_kwargs.setdefault("alpha", 0.1)

    y_mean_plot_kwargs = matplotlib_kwarg_dealiaser(y_mean_plot_kwargs, "plot")
    y_mean_plot_kwargs.setdefault("color", "red")
    y_mean_plot_kwargs.setdefault("linestyle", "dashed")

    vline_kwargs = matplotlib_kwarg_dealiaser(vline_kwargs, "plot")
    vline_kwargs.setdefault("color", "black")
    vline_kwargs.setdefault("linestyle", "dashed")

    for i, ax_i in enumerate(np.ravel(axes)[:length_plotters]):
        y_var_name, _, _, y_plotters_i = y_plotters[i]
        x_var_name, _, _, x_plotters_i = x_plotters[i]

        ax_i.plot(x_plotters_i, y_plotters_i, **y_kwargs)
        ax_i.plot([], label="Actual", **y_kwargs)
        if y_hat_plotters is not None or y_forecasts_plotters is not None:
            ax_i.plot([], label="Fitted", **y_mean_plot_kwargs)
            ax_i.plot([], label="Uncertainty", **y_hat_plot_kwargs)

        ax_i.set_xlabel(x_var_name)
        ax_i.set_ylabel(y_var_name)

        if y_hat_plotters is not None:
            *_, y_hat_plotters_i = y_hat_plotters[i]
            *_, x_hat_plotters_i = x_plotters[i]
            for j in range(num_samples):
                ax_i.plot(x_hat_plotters_i, y_hat_plotters_i[..., j], **y_hat_plot_kwargs)

            *_, x_mean_plotters_i = x_plotters[i]
            *_, y_mean_plotters_i = y_mean_plotters[i]
            ax_i.plot(x_mean_plotters_i, y_mean_plotters_i, **y_mean_plot_kwargs)

        if y_holdout_plotters is not None:
            *_, y_holdout_plotters_i = y_holdout_plotters[i]
            *_, x_holdout_plotters_i = x_holdout_plotters[i]

            ax_i.plot(x_holdout_plotters_i, y_holdout_plotters_i, **y_kwargs)
            ax_i.axvline(x_plotters_i[-1], **vline_kwargs)

        if y_forecasts_plotters is not None:
            *_, y_forecasts_plotters_i = y_forecasts_plotters[i]
            *_, x_forecasts_plotters_i = x_holdout_plotters[i]
            for j in range(num_samples):
                ax_i.plot(
                    x_forecasts_plotters_i, y_forecasts_plotters_i[..., j], **y_hat_plot_kwargs
                )

            *_, x_forecasts_mean_plotters_i = x_holdout_plotters[i]
            *_, y_forecasts_mean_plotters_i = y_forecasts_mean_plotters[i]
            ax_i.plot(
                x_forecasts_mean_plotters_i, y_forecasts_mean_plotters_i, **y_mean_plot_kwargs
            )
            ax_i.axvline(x_plotters_i[-1], **vline_kwargs)

        if legend:
            ax_i.legend(fontsize=xt_labelsize, loc="upper left")

    if backend_show(show):
        plt.show()

    return axes
