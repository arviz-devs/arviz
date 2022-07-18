"""Matplotlib plot linear regression figure."""
import matplotlib.pyplot as plt
import numpy as np

from ...plot_utils import _scale_fig_size
from ...hdiplot import plot_hdi
from . import create_axes_grid, matplotlib_kwarg_dealiaser, backend_show, backend_kwarg_defaults


def plot_lm(
    x,
    y,
    y_model,
    y_hat,
    num_samples,
    kind_pp,
    kind_model,
    xjitter,
    length_plotters,
    rows,
    cols,
    y_kwargs,
    y_hat_plot_kwargs,
    y_hat_fill_kwargs,
    y_model_plot_kwargs,
    y_model_fill_kwargs,
    y_model_mean_kwargs,
    backend_kwargs,
    show,
    figsize,
    textsize,
    axes,
    legend,
    grid,
):
    """Matplotlib Linear Regression."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, _, _, xt_labelsize, _, _ = _scale_fig_size(figsize, textsize, rows, cols)
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs.setdefault("squeeze", False)

    if axes is None:
        _, axes = create_axes_grid(length_plotters, rows, cols, backend_kwargs=backend_kwargs)

    for i, ax_i in enumerate(np.ravel(axes)[:length_plotters]):

        # All the kwargs are defined here beforehand
        y_kwargs = matplotlib_kwarg_dealiaser(y_kwargs, "plot")
        y_kwargs.setdefault("color", "C3")
        y_kwargs.setdefault("marker", ".")
        y_kwargs.setdefault("markersize", 15)
        y_kwargs.setdefault("linewidth", 0)
        y_kwargs.setdefault("zorder", 10)
        y_kwargs.setdefault("label", "observed_data")

        y_hat_plot_kwargs = matplotlib_kwarg_dealiaser(y_hat_plot_kwargs, "plot")
        y_hat_plot_kwargs.setdefault("color", "C1")
        y_hat_plot_kwargs.setdefault("alpha", 0.3)
        y_hat_plot_kwargs.setdefault("markersize", 10)
        y_hat_plot_kwargs.setdefault("marker", ".")
        y_hat_plot_kwargs.setdefault("linewidth", 0)

        y_hat_fill_kwargs = matplotlib_kwarg_dealiaser(y_hat_fill_kwargs, "fill_between")
        y_hat_fill_kwargs.setdefault("color", "C1")

        y_model_plot_kwargs = matplotlib_kwarg_dealiaser(y_model_plot_kwargs, "plot")
        y_model_plot_kwargs.setdefault("color", "k")
        y_model_plot_kwargs.setdefault("alpha", 0.5)
        y_model_plot_kwargs.setdefault("linewidth", 0.5)
        y_model_plot_kwargs.setdefault("zorder", 9)

        y_model_fill_kwargs = matplotlib_kwarg_dealiaser(y_model_fill_kwargs, "fill_between")
        y_model_fill_kwargs.setdefault("color", "k")
        y_model_fill_kwargs.setdefault("linewidth", 0.5)
        y_model_fill_kwargs.setdefault("zorder", 9)
        y_model_fill_kwargs.setdefault("alpha", 0.5)

        y_model_mean_kwargs = matplotlib_kwarg_dealiaser(y_model_mean_kwargs, "plot")
        y_model_mean_kwargs.setdefault("color", "y")
        y_model_mean_kwargs.setdefault("linewidth", 0.8)
        y_model_mean_kwargs.setdefault("zorder", 11)

        y_var_name, _, _, y_plotters = y[i]
        x_var_name, _, _, x_plotters = x[i]
        ax_i.plot(x_plotters, y_plotters, **y_kwargs)
        ax_i.set_xlabel(x_var_name)
        ax_i.set_ylabel(y_var_name)

        if y_hat is not None:
            _, _, _, y_hat_plotters = y_hat[i]
            if kind_pp == "samples":
                for j in range(num_samples):
                    if xjitter is True:
                        jitter_scale = x_plotters[1] - x_plotters[0]
                        scale_high = jitter_scale * 0.2
                        x_plotters_jitter = x_plotters + np.random.uniform(
                            low=-scale_high, high=scale_high, size=len(x_plotters)
                        )
                        ax_i.plot(x_plotters_jitter, y_hat_plotters[..., j], **y_hat_plot_kwargs)
                    else:
                        ax_i.plot(x_plotters, y_hat_plotters[..., j], **y_hat_plot_kwargs)
                ax_i.plot([], **y_hat_plot_kwargs, label="Posterior predictive samples")
            else:
                plot_hdi(x_plotters, y_hat_plotters, ax=ax_i, **y_hat_fill_kwargs)
                ax_i.plot(
                    [], color=y_hat_fill_kwargs["color"], label="Posterior predictive samples"
                )

        if y_model is not None:
            _, _, _, y_model_plotters = y_model[i]
            if kind_model == "lines":
                for j in range(num_samples):
                    ax_i.plot(x_plotters, y_model_plotters[..., j], **y_model_plot_kwargs)
                ax_i.plot([], **y_model_plot_kwargs, label="Uncertainty in mean")

                y_model_mean = np.mean(y_model_plotters, axis=1)
            else:
                plot_hdi(
                    x_plotters,
                    y_model_plotters,
                    fill_kwargs=y_model_fill_kwargs,
                    ax=ax_i,
                )
                ax_i.plot([], color=y_model_fill_kwargs["color"], label="Uncertainty in mean")

                y_model_mean = np.mean(y_model_plotters, axis=(0, 1))
            ax_i.plot(x_plotters, y_model_mean, **y_model_mean_kwargs, label="Mean")

        if legend:
            ax_i.legend(fontsize=xt_labelsize, loc="upper left")
        if grid:
            ax_i.grid(True)

    if backend_show(show):
        plt.show()
    return axes
