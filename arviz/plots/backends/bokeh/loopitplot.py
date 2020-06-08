"""Bokeh loopitplot."""
import bokeh.plotting as bkp
from bokeh.models import BoxAnnotation
import numpy as np

from . import backend_kwarg_defaults
from .. import show_layout
from ...kdeplot import _fast_kde


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
    plot_kwargs,
    backend_kwargs,
    show,
):
    """Bokeh loo pit plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(("dpi", "plot.bokeh.figure.dpi"),),
        **backend_kwargs,
    }
    dpi = backend_kwargs.pop("dpi")
    if ax is None:
        backend_kwargs.setdefault("width", int(figsize[0] * dpi))
        backend_kwargs.setdefault("height", int(figsize[1] * dpi))
        ax = bkp.figure(x_range=(0, 1), **backend_kwargs)

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
            if fill_kwargs.get("drawstyle") == "steps-mid":
                # use step patch when you find out how to do that
                ax.patch(
                    np.concatenate((unif_ecdf, unif_ecdf[::-1])),
                    np.concatenate((p975 - unif_ecdf, (p025 - unif_ecdf)[::-1])),
                    fill_color=fill_kwargs.get("color"),
                    fill_alpha=fill_kwargs.get("alpha", 1.0),
                )
            else:
                ax.patch(
                    np.concatenate((unif_ecdf, unif_ecdf[::-1])),
                    np.concatenate((p975 - unif_ecdf, (p025 - unif_ecdf)[::-1])),
                    fill_color=fill_kwargs.get("color"),
                    fill_alpha=fill_kwargs.get("alpha", 1.0),
                )
        else:
            if fill_kwargs is not None and fill_kwargs.get("drawstyle") == "steps-mid":
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
            ax.add_layout(
                BoxAnnotation(
                    bottom=hdi_odds[1],
                    top=hdi_odds[0],
                    fill_alpha=hdi_kwargs.pop("alpha"),
                    fill_color=hdi_kwargs.pop("color"),
                    **hdi_kwargs
                )
            )
        else:
            for idx in range(n_unif):
                unif_density, xmin, xmax = _fast_kde(unif[idx, :])
                x_s = np.linspace(xmin, xmax, len(unif_density))
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

    show_layout(ax, show)

    return ax
