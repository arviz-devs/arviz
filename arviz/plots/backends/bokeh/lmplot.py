"""Bokeh linear regression plot."""

import numpy as np
from bokeh.models.annotations import Legend

from ...hdiplot import plot_hdi

from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


def plot_lm(
    x,
    y,
    y_model,
    y_hat,
    num_samples,
    kind_pp,
    kind_model,
    length_plotters,
    xjitter,
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
    grid,  # pylint: disable=unused-argument
):
    """Bokeh linreg plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, *_ = _scale_fig_size(figsize, textsize, rows, cols)
    if axes is None:
        axes = create_axes_grid(length_plotters, rows, cols, backend_kwargs=backend_kwargs)

    if y_kwargs is None:
        y_kwargs = {}
    else:
        y_kwargs = y_kwargs.copy()
    y_kwargs.setdefault("marker", "circle")
    y_kwargs.setdefault("fill_color", "red")
    y_kwargs.setdefault("line_width", 0)
    y_kwargs.setdefault("size", 3)

    if y_hat_plot_kwargs is None:
        y_hat_plot_kwargs = {}
    else:
        y_hat_plot_kwargs = y_hat_plot_kwargs.copy()
    y_hat_plot_kwargs.setdefault("marker", "circle")
    y_hat_plot_kwargs.setdefault("fill_color", "orange")
    y_hat_plot_kwargs.setdefault("line_width", 0)

    if y_hat_fill_kwargs is None:
        y_hat_fill_kwargs = {}
    else:
        y_hat_fill_kwargs = y_hat_fill_kwargs.copy()
    # Convert matplotlib color to bokeh fill_color if needed
    if "color" in y_hat_fill_kwargs and "fill_color" not in y_hat_fill_kwargs:
        y_hat_fill_kwargs["fill_color"] = y_hat_fill_kwargs.pop("color")
    y_hat_fill_kwargs.setdefault("fill_color", "orange")
    y_hat_fill_kwargs.setdefault("fill_alpha", 0.5)

    if y_model_plot_kwargs is None:
        y_model_plot_kwargs = {}
    y_model_plot_kwargs.setdefault("line_color", "black")
    y_model_plot_kwargs.setdefault("line_alpha", 0.5)
    y_model_plot_kwargs.setdefault("line_width", 0.5)

    if y_model_fill_kwargs is None:
        y_model_fill_kwargs = {}
    else:
        y_model_fill_kwargs = y_model_fill_kwargs.copy()
    # Convert matplotlib color to bokeh fill_color if needed
    if "color" in y_model_fill_kwargs and "fill_color" not in y_model_fill_kwargs:
        y_model_fill_kwargs["fill_color"] = y_model_fill_kwargs.pop("color")
    y_model_fill_kwargs.setdefault("fill_color", "black")
    y_model_fill_kwargs.setdefault("fill_alpha", 0.5)

    if y_model_mean_kwargs is None:
        y_model_mean_kwargs = {}
    y_model_mean_kwargs.setdefault("line_color", "yellow")
    y_model_mean_kwargs.setdefault("line_width", 2)

    for i, ax_i in enumerate((item for item in axes.flatten() if item is not None)):
        _, _, _, y_plotters = y[i]
        _, _, _, x_plotters = x[i]
        legend_it = []
        observed_legend = ax_i.scatter(x_plotters, y_plotters, **y_kwargs)
        legend_it.append(("Observed", [observed_legend]))

        if y_hat is not None:
            _, _, _, y_hat_plotters = y_hat[i]
            if kind_pp == "samples":
                posterior_legend = []
                for j in range(num_samples):
                    if xjitter is True:
                        jitter_scale = x_plotters[1] - x_plotters[0]
                        scale_high = jitter_scale * 0.2
                        x_plotters_jitter = x_plotters + np.random.uniform(
                            low=-scale_high, high=scale_high, size=len(x_plotters)
                        )
                        posterior_circle = ax_i.scatter(
                            x_plotters_jitter,
                            y_hat_plotters[..., j],
                            alpha=0.2,
                            **y_hat_plot_kwargs,
                        )
                    else:
                        posterior_circle = ax_i.scatter(
                            x_plotters, y_hat_plotters[..., j], alpha=0.2, **y_hat_plot_kwargs
                        )
                    posterior_legend.append(posterior_circle)
                legend_it.append(("Posterior predictive samples", posterior_legend))

            else:
                plot_hdi(
                    x_plotters,
                    y_hat_plotters,
                    ax=ax_i,
                    backend="bokeh",
                    fill_kwargs=y_hat_fill_kwargs,
                    show=False,
                )

        if y_model is not None:
            _, _, _, y_model_plotters = y_model[i]
            if kind_model == "lines":
                model_legend = ax_i.multi_line(
                    [np.tile(np.array(x_plotters, dtype=object), (num_samples, 1))],
                    [np.transpose(y_model_plotters)],
                    **y_model_plot_kwargs,
                )
                legend_it.append(("Uncertainty in mean", [model_legend]))

                y_model_mean = np.mean(y_model_plotters, axis=1)
            else:
                plot_hdi(
                    x_plotters,
                    y_model_plotters,
                    fill_kwargs=y_model_fill_kwargs,
                    ax=ax_i,
                    backend="bokeh",
                    show=False,
                )

                y_model_mean = np.mean(y_model_plotters, axis=(0, 1))
                # Plot mean line across all x values instead of just edges
                mean_legend = ax_i.line(x_plotters, y_model_mean, **y_model_mean_kwargs)
                legend_it.append(("Mean", [mean_legend]))
                continue  # Skip the edge plotting since we plotted full line

            x_plotters_edge = [min(x_plotters), max(x_plotters)]
            y_model_mean_edge = [min(y_model_mean), max(y_model_mean)]
            mean_legend = ax_i.line(x_plotters_edge, y_model_mean_edge, **y_model_mean_kwargs)
            legend_it.append(("Mean", [mean_legend]))

        if legend:
            legend = Legend(
                items=legend_it,
                location="top_left",
                orientation="vertical",
            )
            ax_i.add_layout(legend)
            if textsize is not None:
                ax_i.legend.label_text_font_size = f"{textsize}pt"
            ax_i.legend.click_policy = "hide"

    show_layout(axes, show)
    return axes
