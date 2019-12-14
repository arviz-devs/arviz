"""Bokeh loopitplot."""
import bokeh.plotting as bkp
import numpy as np

from ...hpdplot import plot_hpd
from ...kdeplot import _fast_kde
from ....rcparams import rcParams


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
    use_hpd,
    x_vals,
    unif_densities,
    hpd_kwargs,
    n_unif,
    unif,
    plot_unif_kwargs,
    loo_pit_kde,
    plot_kwargs,
    show,
):
    """Bokeh loo pit plot."""
    if ax is None:
        tools = rcParams["plot.bokeh.tools"]
        output_backend = rcParams["plot.bokeh.output_backend"]
        dpi = rcParams["plot.bokeh.figure.dpi"]
        ax = bkp.figure(
            width=int(figsize[0] * dpi),
            height=int(figsize[1] * dpi),
            output_backend=output_backend,
            tools=tools,
        )

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
        if use_hpd:
            plot_hpd(x_vals, unif_densities, backend="bokeh", ax=ax, show=False, **hpd_kwargs)
        else:
            for idx in range(n_unif):
                unif_density, _, _ = _fast_kde(unif[idx, :], xmin=0, xmax=1)
                ax.line(
                    x_vals,
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

    if show:
        bkp.show(ax, toolbar_location="above")

    return ax
