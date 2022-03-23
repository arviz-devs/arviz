"""Matplotlib loopitplot."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex, to_rgb
from xarray import DataArray

from ....stats.density_utils import kde
from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show, create_axes_grid, matplotlib_kwarg_dealiaser


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
    legend,
    labeller,
    y_hat,
    y,
    color,
    textsize,
    hdi_prob,
    plot_kwargs,
    backend_kwargs,
    show,
):
    """Matplotlib loo pit plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    (figsize, _, _, xt_labelsize, linewidth, _) = _scale_fig_size(figsize, textsize, 1, 1)
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True

    if ax is None:
        _, ax = create_axes_grid(1, backend_kwargs=backend_kwargs)

    plot_kwargs = matplotlib_kwarg_dealiaser(plot_kwargs, "plot")
    plot_kwargs["color"] = to_hex(color)
    plot_kwargs.setdefault("linewidth", linewidth * 1.4)
    if isinstance(y, str):
        label = "LOO-PIT ECDF" if ecdf else "LOO-PIT"
        xlabel = y
    elif isinstance(y, DataArray) and y.name is not None:
        label = "LOO-PIT ECDF" if ecdf else "LOO-PIT"
        xlabel = y.name
    elif isinstance(y_hat, str):
        label = "LOO-PIT ECDF" if ecdf else "LOO-PIT"
        xlabel = y_hat
    elif isinstance(y_hat, DataArray) and y_hat.name is not None:
        label = "LOO-PIT ECDF" if ecdf else "LOO-PIT"
        xlabel = y_hat.name
    else:
        label = "LOO-PIT ECDF" if ecdf else "LOO-PIT"
        xlabel = ""
    xlabel = labeller.var_name_to_str(y)

    plot_kwargs.setdefault("label", label)
    plot_kwargs.setdefault("zorder", 5)

    plot_unif_kwargs = matplotlib_kwarg_dealiaser(plot_unif_kwargs, "plot")
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
            fill_kwargs.setdefault("label", f"{hdi_prob * 100:.3g}% credible interval")
    elif use_hdi:
        if hdi_kwargs is None:
            hdi_kwargs = {}
        hdi_kwargs.setdefault("color", to_hex(hsv_to_rgb(light_color)))
        hdi_kwargs.setdefault("alpha", 0.35)
        hdi_kwargs.setdefault("label", "Uniform HDI")

    if ecdf:
        ax.plot(
            np.hstack((0, loo_pit, 1)), np.hstack((0, loo_pit - loo_pit_ecdf, 0)), **plot_kwargs
        )

        if ecdf_fill:
            ax.fill_between(unif_ecdf, p975 - unif_ecdf, p025 - unif_ecdf, **fill_kwargs)
        else:
            ax.plot(unif_ecdf, p975 - unif_ecdf, unif_ecdf, p025 - unif_ecdf, **plot_unif_kwargs)
    else:
        x_ss = np.empty((n_unif, len(loo_pit_kde)))
        u_dens = np.empty((n_unif, len(loo_pit_kde)))
        if use_hdi:
            ax.axhspan(*hdi_odds, **hdi_kwargs)

            # Adds horizontal reference line
            ax.axhline(1, color="w", zorder=1)
        else:
            for idx in range(n_unif):
                x_s, unif_density = kde(unif[idx, :])
                x_ss[idx] = x_s
                u_dens[idx] = unif_density
            ax.plot(x_ss.T, u_dens.T, **plot_unif_kwargs)
        ax.plot(x_vals, loo_pit_kde, **plot_kwargs)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
    ax.set_xlabel(xlabel)
    ax.tick_params(labelsize=xt_labelsize)
    if legend:
        if not (use_hdi or (ecdf and ecdf_fill)):
            label = f"{hdi_prob * 100:.3g}% credible interval" if ecdf else "Uniform"
            ax.plot([], label=label, **plot_unif_kwargs)
        ax.legend()

    if backend_show(show):
        plt.show()

    return ax
