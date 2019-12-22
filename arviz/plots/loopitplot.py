"""Plot LOO-PIT predictive checks of inference data."""
import numpy as np
import scipy.stats as stats
from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb, to_hex
from xarray import DataArray

from ..stats import loo_pit as _loo_pit
from .plot_utils import _scale_fig_size, get_plotting_function
from .kdeplot import _fast_kde


def plot_loo_pit(
    idata=None,
    y=None,
    y_hat=None,
    log_weights=None,
    ecdf=False,
    ecdf_fill=True,
    n_unif=100,
    use_hpd=False,
    credible_interval=0.94,
    figsize=None,
    textsize=None,
    color="C0",
    legend=True,
    ax=None,
    plot_kwargs=None,
    plot_unif_kwargs=None,
    hpd_kwargs=None,
    fill_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
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
        Plot the difference between the LOO-PIT Empirical Cumulative Distribution Function
        (ECDF) and the uniform CDF instead of LOO-PIT kde.
        In this case, instead of overlaying uniform distributions, the beta ``credible_interval``
        interval around the theoretical uniform CDF is shown. This approximation only holds
        for large S and ECDF values not vary close to 0 nor 1. For more information, see
        `Vehtari et al. (2019)`, `Appendix G <https://avehtari.github.io/rhat_ess/rhat_ess.html>`_.
    ecdf_fill : bool, optional
        Use fill_between to mark the area inside the credible interval. Otherwise, plot the
        border lines.
    n_unif : int, optional
        Number of datasets to simulate and overlay from the uniform distribution.
    use_hpd : bool, optional
        Use plot_hpd to fill between hpd values instead of overlaying the uniform distributions.
    credible_interval : float, optional
        Credible interval of the hpd or of the ECDF theoretical credible interval
    figsize : figure size tuple, optional
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int, optional
        Text size for labels. If None it will be autoscaled based on figsize.
    color : str or array_like, optional
        Color of the LOO-PIT estimated pdf plot. If ``plot_unif_kwargs`` has no "color" key,
        an slightly lighter color than this argument will be used for the uniform kde lines.
        This will ensure that LOO-PIT kde and uniform kde have different default colors.
    legend : bool, optional
        Show the legend of the figure.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    plot_kwargs : dict, optional
        Additional keywords passed to ax.plot for LOO-PIT line (kde or ECDF)
    plot_unif_kwargs : dict, optional
        Additional keywords passed to ax.plot for overlaid uniform distributions or
        for beta credible interval lines if ``ecdf=True``
    hpd_kwargs : dict, optional
        Additional keywords passed to az.plot_hpd
    fill_kwargs : dict, optional
        Additional kwargs passed to ax.fill_between
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : matplotlib axes or bokeh figures

    References
    ----------
    * Gabry et al. (2017) see https://arxiv.org/abs/1709.01449
    * https://mc-stan.org/bayesplot/reference/PPC-loo.html
    * Gelman et al. BDA (2014) Section 6.3

    Examples
    --------
    Plot LOO-PIT predictive checks overlaying the KDE of the LOO-PIT values to several
    realizations of uniform variable sampling with the same number of observations.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> idata = az.load_arviz_data("centered_eight")
        >>> az.plot_loo_pit(idata=idata, y="obs")

    Fill the area containing the 94% credible interval of the difference between uniform
    variables empirical CDF and the real uniform CDF. A LOO-PIT ECDF clearly outside of these
    theoretical boundaries indicates that the observations and the posterior predictive
    samples do not follow the same distribution.

    .. plot::
        :context: close-figs

        >>> az.plot_loo_pit(idata=idata, y="obs", ecdf=True)

    """
    if ecdf and use_hpd:
        raise ValueError("use_hpd is incompatible with ecdf plot")

    (figsize, _, _, xt_labelsize, linewidth, _) = _scale_fig_size(figsize, textsize, 1, 1)

    loo_pit = _loo_pit(idata=idata, y=y, y_hat=y_hat, log_weights=log_weights)
    loo_pit = loo_pit.flatten() if isinstance(loo_pit, np.ndarray) else loo_pit.values.flatten()

    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs["color"] = to_hex(color)
    plot_kwargs.setdefault("linewidth", linewidth * 1.4)
    if isinstance(y, str):
        label = ("{} LOO-PIT ECDF" if ecdf else "{} LOO-PIT").format(y)
    elif isinstance(y, DataArray):
        label = ("{} LOO-PIT ECDF" if ecdf else "{} LOO-PIT").format(y.name)
    elif isinstance(y_hat, str):
        label = ("{} LOO-PIT ECDF" if ecdf else "{} LOO-PIT").format(y_hat)
    elif isinstance(y_hat, DataArray):
        label = ("{} LOO-PIT ECDF" if ecdf else "{} LOO-PIT").format(y_hat.name)
    else:
        label = "LOO-PIT ECDF" if ecdf else "LOO-PIT"

    plot_kwargs.setdefault("label", label)
    plot_kwargs.setdefault("zorder", 5)

    if plot_unif_kwargs is None:
        plot_unif_kwargs = {}
    light_color = rgb_to_hsv(to_rgb(plot_kwargs.get("color")))
    light_color[1] /= 2  # pylint: disable=unsupported-assignment-operation
    light_color[2] += (1 - light_color[2]) / 2  # pylint: disable=unsupported-assignment-operation
    plot_unif_kwargs.setdefault("color", to_hex(hsv_to_rgb(light_color)))
    plot_unif_kwargs.setdefault("alpha", 0.5)
    plot_unif_kwargs.setdefault("linewidth", 0.6 * linewidth)

    loo_pit_ecdf = None
    unif_ecdf = None
    p975 = None
    p025 = None
    loo_pit_kde = None
    unif = None
    unif_densities = None
    x_vals = None

    if ecdf:
        loo_pit.sort()
        n_data_points = loo_pit.size
        loo_pit_ecdf = np.arange(n_data_points) / n_data_points
        # ideal unnormalized ECDF of uniform distribution with n_data_points points
        # it is used indistinctively as x or p(u<x) because for u~U(0,1) they are equal
        unif_ecdf = np.arange(n_data_points + 1)
        p975 = stats.beta.ppf(
            0.5 + credible_interval / 2, unif_ecdf + 1, n_data_points - unif_ecdf + 1
        )
        p025 = stats.beta.ppf(
            0.5 - credible_interval / 2, unif_ecdf + 1, n_data_points - unif_ecdf + 1
        )
        unif_ecdf = unif_ecdf / n_data_points

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
            fill_kwargs.setdefault("label", "{:.3g}% credible interval".format(credible_interval))
    else:
        loo_pit_kde, _, _ = _fast_kde(loo_pit, xmin=0, xmax=1)

        unif = np.random.uniform(size=(n_unif, loo_pit.size))
        x_vals = np.linspace(0, 1, len(loo_pit_kde))
        if use_hpd:
            if hpd_kwargs is None:
                hpd_kwargs = {}
            hpd_kwargs.setdefault("color", to_hex(hsv_to_rgb(light_color)))
            hpd_fill_kwargs = hpd_kwargs.pop("fill_kwargs", {})
            hpd_fill_kwargs.setdefault("label", "Uniform HPD")
            hpd_kwargs["fill_kwargs"] = hpd_fill_kwargs
            hpd_kwargs["credible_interval"] = credible_interval

            unif_densities = np.empty((n_unif, len(loo_pit_kde)))

    loo_pit_kwargs = dict(
        ax=ax,
        figsize=figsize,
        ecdf=ecdf,
        loo_pit=loo_pit,
        loo_pit_ecdf=loo_pit_ecdf,
        unif_ecdf=unif_ecdf,
        p975=p975,
        p025=p025,
        fill_kwargs=fill_kwargs,
        ecdf_fill=ecdf_fill,
        use_hpd=use_hpd,
        x_vals=x_vals,
        unif_densities=unif_densities,
        hpd_kwargs=hpd_kwargs,
        n_unif=n_unif,
        unif=unif,
        plot_unif_kwargs=plot_unif_kwargs,
        loo_pit_kde=loo_pit_kde,
        xt_labelsize=xt_labelsize,
        legend=legend,
        credible_interval=credible_interval,
        plot_kwargs=plot_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend == "bokeh":

        if (
            loo_pit_kwargs["hpd_kwargs"] is not None
            and "fill_kwargs" in loo_pit_kwargs["hpd_kwargs"]
            and loo_pit_kwargs["hpd_kwargs"]["fill_kwargs"] is not None
            and "label" in loo_pit_kwargs["hpd_kwargs"]["fill_kwargs"]
        ):
            loo_pit_kwargs["hpd_kwargs"]["fill_kwargs"].pop("label")
        loo_pit_kwargs.pop("legend")
        loo_pit_kwargs.pop("xt_labelsize")
        loo_pit_kwargs.pop("credible_interval")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_loo_pit", "loopitplot", backend)
    axes = plot(**loo_pit_kwargs)

    return axes
