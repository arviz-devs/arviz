"""Plot LOO-PIT predictive checks of inference data."""
import numpy as np
import scipy.stats as stats

from ..stats import loo_pit as _loo_pit
from ..numeric_utils import _fast_kde
from .plot_utils import get_plotting_function

from ..rcparams import rcParams


def plot_loo_pit(
    idata=None,
    y=None,
    y_hat=None,
    log_weights=None,
    ecdf=False,
    ecdf_fill=True,
    n_unif=100,
    use_hdi=False,
    credible_interval=None,
    figsize=None,
    textsize=None,
    color="C0",
    legend=True,
    ax=None,
    plot_kwargs=None,
    plot_unif_kwargs=None,
    hdi_kwargs=None,
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
    use_hdi : bool, optional
        Compute expected hdi values instead of overlaying the sampled uniform distributions.
    credible_interval : float, optional
        Theoretical credible interval. Works with ``use_hdi=True`` or ``ecdf=True``.
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
    hdi_kwargs : dict, optional
        Additional keywords passed to ax.axhspan
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
        >>> idata = az.load_arviz_data("radon")
        >>> az.plot_loo_pit(idata=idata, y="y")

    Fill the area containing the 94% highest density interval of the difference between uniform
    variables empirical CDF and the real uniform CDF. A LOO-PIT ECDF clearly outside of these
    theoretical boundaries indicates that the observations and the posterior predictive
    samples do not follow the same distribution.

    .. plot::
        :context: close-figs

        >>> az.plot_loo_pit(idata=idata, y="y", ecdf=True)

    """
    if ecdf and use_hdi:
        raise ValueError("use_hdi is incompatible with ecdf plot")

    loo_pit = _loo_pit(idata=idata, y=y, y_hat=y_hat, log_weights=log_weights)
    loo_pit = loo_pit.flatten() if isinstance(loo_pit, np.ndarray) else loo_pit.values.flatten()

    loo_pit_ecdf = None
    unif_ecdf = None
    p975 = None
    p025 = None
    loo_pit_kde = None
    hdi_odds = None
    unif = None
    x_vals = None

    if credible_interval is None:
        credible_interval = rcParams["stats.hdi_prob"]
    else:
        if not 1 >= credible_interval > 0:
            raise ValueError("The value of credible_interval should be in the interval (0, 1]")

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
    else:
        loo_pit_kde, xmin, xmax = _fast_kde(loo_pit)

        unif = np.random.uniform(size=(n_unif, loo_pit.size))
        x_vals = np.linspace(xmin, xmax, len(loo_pit_kde))
        if use_hdi:
            n_obs = loo_pit.size
            hdi_ = stats.beta(n_obs / 2, n_obs / 2).ppf((1 - credible_interval) / 2)
            hdi_odds = (hdi_ / (1 - hdi_), (1 - hdi_) / hdi_)

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
        use_hdi=use_hdi,
        x_vals=x_vals,
        hdi_kwargs=hdi_kwargs,
        hdi_odds=hdi_odds,
        n_unif=n_unif,
        unif=unif,
        plot_unif_kwargs=plot_unif_kwargs,
        loo_pit_kde=loo_pit_kde,
        textsize=textsize,
        color=color,
        legend=legend,
        y_hat=y_hat,
        y=y,
        credible_interval=credible_interval,
        plot_kwargs=plot_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_loo_pit", "loopitplot", backend)
    axes = plot(**loo_pit_kwargs)

    return axes
