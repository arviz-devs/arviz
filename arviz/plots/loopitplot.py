"""Plot LOO-PIT predictive checks of inference data."""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb

from ..stats import loo_pit as _loo_pit
from .plot_utils import _scale_fig_size
from .kdeplot import _fast_kde
from .hpdplot import plot_hpd


def plot_loo_pit(
    idata=None,
    y=None,
    y_hat=None,
    log_weights=None,
    ecdf=False,
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
        InferenceData object.
    y : array, DataArray or str
        Observed data. If str, idata must be present and contain the observed data group
    y_hat : array, DataArray or str
        Posterior predictive samples for ``y``. It must have the same shape as y plus an
        extra dimension at the end of size n_samples (chains and draws stacked). If str or
        None, idata must contain the posterior predictive group. If None, y_hat is taken
        equal to y, thus, y must be str too.
    log_weights : array or DataArray
        Smoothed log_weights. It must have the same shape as ``y_hat``
    ecdf : bool, optional
        Plot LOO-PIT Empirical Cumulative Distribution Function (ECDF) instead of LOO-PIT kde.
        In this case, instead of overlaying uniform distributions, the beta 95% interval around
        the theoretical uniform CDF is shown. For more information, see `Vehtari et al. (2019)`,
        Appendix G.
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
        Additional keywords passed to ax.plot for LOO-PIT line (kde or ECDF)
    plot_unif_kwargs : dict, optional
        Additional keywords passed to ax.plot for overlaid uniform distributions or
        for beta 95% interval lines if ``ecdf=True``
    hpd_kwargs : dict, optional
        Additional keywords passed to az.plot_hpd

    Returns
    -------
    axes : axes
        Matplotlib axes
    """
    if ecdf and use_hpd:
        raise ValueError("use_hpd is incompatible with ecdf plot")

    (figsize, _, _, xt_labelsize, linewidth, _) = _scale_fig_size(figsize, textsize, 1, 1)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs["color"] = color
    plot_kwargs.setdefault("linewidth", linewidth * 1.4)
    plot_kwargs.setdefault("label", "LOO-PIT ECDF" if ecdf else "LOO-PIT")
    plot_kwargs.setdefault("zorder", 5)

    if plot_unif_kwargs is None:
        plot_unif_kwargs = {}
    light_color = rgb_to_hsv(to_rgb(plot_kwargs.get("color")))
    light_color[1] /= 2
    light_color[2] += (1 - light_color[2]) / 2
    plot_unif_kwargs.setdefault("color", hsv_to_rgb(light_color))
    plot_unif_kwargs.setdefault("alpha", 0.5)
    plot_unif_kwargs.setdefault("linewidth", 0.6 * linewidth)

    if hpd_kwargs is None:
        hpd_kwargs = {}
    hpd_kwargs.setdefault("color", hsv_to_rgb(light_color))
    fill_kwargs = hpd_kwargs.pop("fill_kwargs", {})
    fill_kwargs.setdefault("label", "Uniform HPD")
    hpd_kwargs["fill_kwargs"] = fill_kwargs

    loo_pit = _loo_pit(idata, y, y_hat, log_weights)
    loo_pit = loo_pit.flatten() if isinstance(loo_pit, np.ndarray) else loo_pit.values.flatten()

    if ecdf:
        loo_pit.sort()
        n_data_points = loo_pit.size
        loo_pit_ecdf = np.arange(n_data_points) / n_data_points
        # ideal unnormalized ECDF of uniform distribution with n_data_points points
        unif_ecdf = np.arange(n_data_points + 1)
        p975 = stats.beta.ppf(0.975, unif_ecdf + 1, n_data_points - unif_ecdf + 1)
        p025 = stats.beta.ppf(0.025, unif_ecdf + 1, n_data_points - unif_ecdf + 1)

        plot_kwargs.setdefault("drawstyle", "steps" if n_data_points < 100 else "default")
        plot_unif_kwargs.setdefault("drawstyle", "steps" if n_data_points < 100 else "default")

        ax.plot(np.hstack((0, loo_pit, 1)), np.hstack((0, loo_pit_ecdf, 1)), **plot_kwargs)
        ax.plot(
            unif_ecdf / n_data_points, p975, unif_ecdf / n_data_points, p025, **plot_unif_kwargs
        )
    else:
        loo_pit_kde, _, _ = _fast_kde(loo_pit, xmin=0, xmax=1)

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

    ax.tick_params(labelsize=xt_labelsize)
    if legend:
        if not use_hpd:
            ax.plot([], label="Uniform", **plot_unif_kwargs)
        ax.legend()
    return ax
