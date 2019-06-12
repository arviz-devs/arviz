import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb

from ..data import InferenceData
from ..stats import loo_pit as _loo_pit, psislw as _psislw
from .plot_utils import _scale_fig_size
from .kdeplot import _fast_kde
from .hpdplot import plot_hpd


def plot_loo_pit(
    idata,
    y,
    y_hat=None,
    n_unif=100,
    use_hpd=False,
    figsize=None,
    textsize=None,
    color="C0",
    legend=True,
    ax=None,
    plot_kwargs=None,
    plot_unif_kwargs=None,
    hpd_kwargs=None,
):
    """Plot Leave-One-Out (LOO) probability integral transformation (PIT) predictive checks.

    Parameters
    ----------
    idata : InferenceData
        InferenceData object with groups `observed_data`, `posterior_predictive` and
        `sample_stats`. Objects that can be converted to InferenceData are not accepted.
    y : str
        Name of the observed_data variable to use.
    y_hat : str, optional
        Name of the posterior_predictive variable to use. If None, it will be taken as equal
        to y.
    n_unif : int, optional
        Number of datasets to simulate and overlay from the uniform distribution.
    use_hpd : bool, optional
        Use plot_hpd to fill between hpd values instead of overlaying the uniform distributions.
    figsize : figure size tuple, optional
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int, optional
        Text size for labels. If None it will be autoscaled based on figsize.
    color : str or array_like, optional
        Color of the LOO-PIT estimated pdf plot.
    legend : bool, optional
        Show the legend of the figure.
    ax : axes, optional
        Matplotlib axes
    plot_kwargs : dict, optional
        Additional keywords passed to ax.plot, for LOO-PIT line
    plot_unif_kwargs : dict, optional
        Additional keywords passed to ax.plot, for overlaid uniform distributions.
    hpd_kwargs : dict, optional
        Additional keywords passed to az.plot_hpd

    Returns
    -------
    axes : axes
        Matplotlib axes
    """
    if not isinstance(idata, InferenceData):
        raise ValueError("idata must be of type InferenceData")

    if ax is None:
        (figsize, ax_labelsize, _, xt_labelsize, linewidth, markersize) = _scale_fig_size(
            figsize, textsize, 1, 1
        )
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs["color"] = color
    plot_kwargs.setdefault("linewidth", linewidth)
    plot_kwargs.setdefault("label", "LOO-PIT")

    if plot_unif_kwargs is None:
        plot_unif_kwargs = {}
    light_color = rgb_to_hsv(to_rgb(plot_kwargs.get("color")))
    light_color[1] /= 2
    light_color[2] += (1 - light_color[2]) / 2
    plot_unif_kwargs.setdefault("color", hsv_to_rgb(light_color))
    plot_unif_kwargs.setdefault("alpha", 0.5)
    plot_unif_kwargs.setdefault("linewidth", 0.6 * linewidth)
    plot_unif_kwargs.setdefault("zorder", -1)

    if hpd_kwargs is None:
        hpd_kwargs = {}
    hpd_kwargs.setdefault("color", hsv_to_rgb(light_color))
    fill_kwargs = hpd_kwargs.pop("fill_kwargs", {})
    fill_kwargs.setdefault("label", "Uniform HPD")
    hpd_kwargs["fill_kwargs"] = fill_kwargs

    if y_hat is None:
        y_hat = y
    y = idata.observed_data[y]
    y_hat = idata.posterior_predictive[y_hat].stack(samples=("chain", "draw"))
    log_likelihood = idata.sample_stats.log_likelihood.stack(samples=("chain", "draw"))
    log_weights, _ = _psislw(-log_likelihood)

    if log_weights.dims == y_hat.dims and y_hat.dims[:-1] == y.dims:
        loo_pit = _loo_pit(y, y_hat, log_weights).values
    else:
        loo_pit = _loo_pit(y.values, y_hat.values, log_weights.values)
    loo_pit_kde, _, _ = _fast_kde(loo_pit.flatten(), xmin=0, xmax=1)

    unif = np.random.uniform(size=(n_unif, loo_pit.size))
    x_vals = np.linspace(0, 1, len(loo_pit_kde))
    if use_hpd:
        unif_densities = np.empty((n_unif, len(loo_pit_kde)))
        for idx in range(n_unif):
            unif_densities[idx, :], _, _ = _fast_kde(unif[idx, :], xmin=0, xmax=1)
        plot_hpd(x_vals, unif_densities, **hpd_kwargs)
    else:
        for idx in range(n_unif):
            unif_density, _, _ = _fast_kde(unif[idx, :], xmin=0, xmax=1)
            ax.plot(x_vals, unif_density, **plot_unif_kwargs)
    ax.plot(x_vals, loo_pit_kde, **plot_kwargs)

    if legend:
        if not use_hpd:
            ax.plot([], label="Uniform", **plot_unif_kwargs)
        ax.legend()
    return ax
