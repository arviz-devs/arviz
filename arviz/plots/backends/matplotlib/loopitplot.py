"""Matplotlib loopitplot."""
import matplotlib.pyplot as plt
import numpy as np

from . import backend_kwarg_defaults, backend_show
from ....numeric_utils import _fast_kde


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
    xt_labelsize,
    legend,
    credible_interval,
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
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize, **backend_kwargs)

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
        else:
            for idx in range(n_unif):
                unif_density, xmin, xmax = _fast_kde(unif[idx, :])
                x_s = np.linspace(xmin, xmax, len(unif_density))
                x_ss[idx] = x_s
                u_dens[idx] = unif_density
            ax.plot(x_ss.T, u_dens.T, **plot_unif_kwargs)
        ax.plot(x_vals, loo_pit_kde, **plot_kwargs)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
    ax.tick_params(labelsize=xt_labelsize)
    if legend:
        if not (use_hdi or (ecdf and ecdf_fill)):
            label = "{:.3g}% credible interval".format(credible_interval) if ecdf else "Uniform"
            ax.plot([], label=label, **plot_unif_kwargs)
        ax.legend()

    if backend_show(show):
        plt.show()

    return ax
