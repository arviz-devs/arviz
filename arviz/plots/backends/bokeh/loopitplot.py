"""Bokeh loopitplot."""
import numpy as np
from bokeh.models import BoxAnnotation
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb
from xarray import DataArray

from ....stats.density_utils import kde
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


def plot_loo_pit(
    ax,
    figsize,
    ecdf,
    loo_pit,
    loo_pit_ecdf,
    unif_ecdf,
    p975,
    p025,
    fill_kwargs,
    ecdf_fill,
    use_hdi,
    x_vals,
    hdi_kwargs,
    hdi_odds,
    n_unif,
    unif,
    plot_unif_kwargs,
    loo_pit_kde,
    legend,  # pylint: disable=unused-argument
    y_hat,
    y,
    color,
    textsize,
    labeller,
    hdi_prob,
    plot_kwargs,
    backend_kwargs,
    show,
):
    """Bokeh loo pit plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    (figsize, *_, linewidth, _) = _scale_fig_size(figsize, textsize, 1, 1)

    if ax is None:
        backend_kwargs.setdefault("x_range", (0, 1))
        ax = create_axes_grid(
            1,
            figsize=figsize,
            squeeze=True,
            backend_kwargs=backend_kwargs,
        )

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    plot_kwargs.setdefault("color", to_hex(color))
    plot_kwargs.setdefault("linewidth", linewidth * 1.4)
    if isinstance(y, str):
        xlabel = y
    elif isinstance(y, DataArray) and y.name is not None:
        xlabel = y.name
    elif isinstance(y_hat, str):
        xlabel = y_hat
    elif isinstance(y_hat, DataArray) and y_hat.name is not None:
        xlabel = y_hat.name
    else:
        xlabel = ""
    label = "LOO-PIT ECDF" if ecdf else "LOO-PIT"
    xlabel = labeller.var_name_to_str(xlabel)

    plot_kwargs.setdefault("legend_label", label)

    plot_unif_kwargs = {} if plot_unif_kwargs is None else plot_unif_kwargs
    light_color = rgb_to_hsv(to_rgb(plot_kwargs.get("color")))
    light_color[1] /= 2  # pylint: disable=unsupported-assignment-operation
    light_color[2] += (1 - light_color[2]) / 2  # pylint: disable=unsupported-assignment-operation
    plot_unif_kwargs.setdefault("color", to_hex(hsv_to_rgb(light_color)))
    plot_unif_kwargs.setdefault("alpha", 0.5)
    plot_unif_kwargs.setdefault("linewidth", 0.6 * linewidth)

    if ecdf:
        n_data_points = loo_pit.size
        plot_kwargs.setdefault("drawstyle", "steps-mid" if n_data_points < 100 else "default")
        plot_unif_kwargs.setdefault("drawstyle", "steps-mid" if n_data_points < 100 else "default")

        if ecdf_fill:
            if fill_kwargs is None:
                fill_kwargs = {}
            fill_kwargs.setdefault("color", to_hex(hsv_to_rgb(light_color)))
            fill_kwargs.setdefault("alpha", 0.5)
            fill_kwargs.setdefault(
                "step", "mid" if plot_kwargs["drawstyle"] == "steps-mid" else None
            )
            fill_kwargs.setdefault("legend_label", f"{hdi_prob * 100:.3g}% credible interval")
    elif use_hdi:
        if hdi_kwargs is None:
            hdi_kwargs = {}
        hdi_kwargs.setdefault("color", to_hex(hsv_to_rgb(light_color)))
        hdi_kwargs.setdefault("alpha", 0.35)

    if ecdf:
        if plot_kwargs.get("drawstyle") == "steps-mid":
            ax.step(
                np.hstack((0, loo_pit, 1)),
                np.hstack((0, loo_pit - loo_pit_ecdf, 0)),
                line_color=plot_kwargs.get("color", "black"),
                line_alpha=plot_kwargs.get("alpha", 1.0),
                line_width=plot_kwargs.get("linewidth", 3.0),
                mode="center",
            )
        else:
            ax.line(
                np.hstack((0, loo_pit, 1)),
                np.hstack((0, loo_pit - loo_pit_ecdf, 0)),
                line_color=plot_kwargs.get("color", "black"),
                line_alpha=plot_kwargs.get("alpha", 1.0),
                line_width=plot_kwargs.get("linewidth", 3.0),
            )

        if ecdf_fill:
            if (
                fill_kwargs.get("drawstyle") == "steps-mid"
                or fill_kwargs.get("drawstyle") != "steps-mid"
            ):
                # use step patch when you find out how to do that
                ax.patch(
                    np.concatenate((unif_ecdf, unif_ecdf[::-1])),
                    np.concatenate((p975 - unif_ecdf, (p025 - unif_ecdf)[::-1])),
                    fill_color=fill_kwargs.get("color"),
                    fill_alpha=fill_kwargs.get("alpha", 1.0),
                )
        elif fill_kwargs is not None and fill_kwargs.get("drawstyle") == "steps-mid":
            ax.step(
                unif_ecdf,
                p975 - unif_ecdf,
                line_color=plot_unif_kwargs.get("color", "black"),
                line_alpha=plot_unif_kwargs.get("alpha", 1.0),
                line_width=plot_kwargs.get("linewidth", 1.0),
                mode="center",
            )
            ax.step(
                unif_ecdf,
                p025 - unif_ecdf,
                line_color=plot_unif_kwargs.get("color", "black"),
                line_alpha=plot_unif_kwargs.get("alpha", 1.0),
                line_width=plot_unif_kwargs.get("linewidth", 1.0),
                mode="center",
            )
        else:
            ax.line(
                unif_ecdf,
                p975 - unif_ecdf,
                line_color=plot_unif_kwargs.get("color", "black"),
                line_alpha=plot_unif_kwargs.get("alpha", 1.0),
                line_width=plot_unif_kwargs.get("linewidth", 1.0),
            )
            ax.line(
                unif_ecdf,
                p025 - unif_ecdf,
                line_color=plot_unif_kwargs.get("color", "black"),
                line_alpha=plot_unif_kwargs.get("alpha", 1.0),
                line_width=plot_unif_kwargs.get("linewidth", 1.0),
            )
    else:
        if use_hdi:
            patch = BoxAnnotation(
                bottom=hdi_odds[1],
                top=hdi_odds[0],
                fill_alpha=hdi_kwargs.pop("alpha"),
                fill_color=hdi_kwargs.pop("color"),
                **hdi_kwargs,
            )
            patch.level = "underlay"
            ax.add_layout(patch)

            # Adds horizontal reference line
            ax.line([0, 1], [1, 1], line_color="white", line_width=1.5)
        else:
            for idx in range(n_unif):
                x_s, unif_density = kde(unif[idx, :])
                ax.line(
                    x_s,
                    unif_density,
                    line_color=plot_unif_kwargs.get("color", "black"),
                    line_alpha=plot_unif_kwargs.get("alpha", 0.1),
                    line_width=plot_unif_kwargs.get("linewidth", 1.0),
                )
        ax.line(
            x_vals,
            loo_pit_kde,
            line_color=plot_kwargs.get("color", "black"),
            line_alpha=plot_kwargs.get("alpha", 1.0),
            line_width=plot_kwargs.get("linewidth", 3.0),
        )

    # Sets xlim(0, 1)
    ax.xaxis.axis_label = xlabel
    ax.line(0, 0)
    ax.line(1, 0)
    show_layout(ax, show)

    return ax
