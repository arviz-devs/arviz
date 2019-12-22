"""Matplotlib loopitplot."""
import matplotlib.pyplot as plt
import numpy as np

from . import backend_kwarg_defaults, backend_show
from ...kdeplot import _fast_kde
from ...hpdplot import plot_hpd


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
        if use_hpd:
            plot_hpd(x_vals, unif_densities, **hpd_kwargs)
        else:
            for idx in range(n_unif):
                unif_density, _, _ = _fast_kde(unif[idx, :], xmin=0, xmax=1)
                ax.plot(x_vals, unif_density, **plot_unif_kwargs)
        ax.plot(x_vals, loo_pit_kde, **plot_kwargs)

    ax.tick_params(labelsize=xt_labelsize)
    if legend:
        if not (use_hpd or (ecdf and ecdf_fill)):
            label = "{:.3g}% credible interval".format(credible_interval) if ecdf else "Uniform"
            ax.plot([], label=label, **plot_unif_kwargs)
        ax.legend()

    if backend_show(show):
        plt.show()

    return ax
