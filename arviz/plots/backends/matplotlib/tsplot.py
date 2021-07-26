"""Matplotlib plot time series figure."""
import matplotlib.pyplot as plt
import numpy as np

from ...plot_utils import _scale_fig_size
from . import create_axes_grid, backend_show, backend_kwarg_defaults


def plot_ts(
    x_plotters,
    y_plotters,
    components_plotters,
    comp_uncertainty_plotters,
    y_hat_plotters,
    y_hat_mean_plotters,
    num_samples,
    holdout,
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

    x_name, _, _, x_plotters = x_plotters[-1]
    x_after_holdout = x_plotters[-holdout:]

    for i, ax_i in enumerate(np.ravel(axes)[:rows]):

        ax_i.set_xlabel(x_name)
        if i == rows - 1 and y_plotters is not None:
            y_name, _, _, y_plotters = y_plotters[-1]
            _, _, _, y_hat_plotters = y_hat_plotters[-1]

            ax_i.plot(x_plotters, y_plotters, zorder=10)
            for j in range(num_samples):
                ax_i.plot(x_after_holdout, y_hat_plotters[..., j], color="grey", alpha=0.1)

            ax_i.set_title(y_name)

            ax_i.plot(
                x_plotters, y_hat_mean_plotters, color="red", linestyle="dashed", label="fitted"
            )
            ax_i.axvline(x_plotters[-holdout], linestyle="dashed", color="black")

            continue

        component_name, _, _, components_plotters_i = components_plotters[i]
        _, _, _, comp_uncertainty_plotters_i = comp_uncertainty_plotters[i]
        ax_i.plot(x_plotters, components_plotters_i, zorder=10)

        ax_i.set_title(component_name)

        for j in range(num_samples):
            ax_i.plot(
                x_after_holdout,
                comp_uncertainty_plotters_i[..., j],
                color="grey",
                marker=".",
                markersize=1,
                alpha=0.02,
            )

        ax_i.axvline(x_plotters[-holdout], linestyle="dashed", color="black")

    if backend_show(show):
        plt.show()

    return axes
