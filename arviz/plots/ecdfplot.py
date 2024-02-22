"""Plot ecdf or ecdf-difference plot with confidence bands."""
import warnings

import numpy as np
from scipy.stats import uniform

from ..rcparams import rcParams
from ..stats.ecdf_utils import compute_ecdf, ecdf_confidence_band, _get_ecdf_points
from .plot_utils import get_plotting_function


def plot_ecdf(
    values,
    values2=None,
    cdf=None,
    difference=False,
    pit=False,
    npoints=100,
    band_kind=None,
    band_prob=None,
    num_trials=500,
    rvs=None,
    random_state=None,
    figsize=None,
    fill_band=True,
    plot_kwargs=None,
    fill_kwargs=None,
    plot_outline_kwargs=None,
    ax=None,
    show=None,
    backend=None,
    backend_kwargs=None,
    confidence_bands=None,
    pointwise=False,
    fpr=None,
    **kwargs,
):
    r"""Plot ECDF or ECDF-Difference Plot with Confidence bands.

    Plots of the empirical CDF estimates of an array. When `values2` argument is provided,
    the two empirical CDFs are overlaid with the distribution of `values` on top
    (in a darker shade) and confidence bands in a more transparent shade. Optionally, the difference
    between the two empirical CDFs can be computed, and the PIT for a single dataset or a comparison
    between two samples.

    Notes
    -----
    This plot computes the confidence bands with the simulated based algorithm presented in [1]_.

    Parameters
    ----------
    values : array-like
        Values to plot from an unknown continuous or discrete distribution.
    values2 : array-like, optional
        Values to compare to the original sample.
    cdf : callable, optional
        Cumulative distribution function of the distribution to compare the original sample.
        The function must take as input a numpy array of draws from the distribution.
    difference : bool, default False
        If True then plot ECDF-difference plot otherwise ECDF plot.
    pit : bool, default False
        If True plots the ECDF or ECDF-diff of PIT of sample.
    band_kind : str, optional
        - None: No confidence bands are plotted.
        - "pointwise": Compute the pointwise (i.e. marginal) confidence band.
        - "simulated": Use Monte Carlo simulation to estimate a simultaneous confidence band.
    band_prob : float, default 0.94
        The probability that the true ECDF lies within the confidence band. If `band_kind` is
        "pointwise", this is the marginal probability instead of the joint probability.
    npoints : int, default 100
        This denotes the granularity size of our plot i.e the number of evaluation points
        for the ecdf or ecdf-difference plots.
    rvs: callable, optional
        A function that takes an integer `ndraws` and optionally the object passed to
        `random_state` and returns an array of `ndraws` samples from the same distribution
        as the original dataset. Required if `method` is "simulated" and variable is discrete.
    num_trials : int, default 500
        The number of random ECDFs to generate for constructing simultaneous confidence bands
        (if `band_kind` is "simulated").
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional
        If `None`, the `numpy.random.RandomState` singleton is used. If an `int`, a new
        ``numpy.random.RandomState`` instance is used, seeded with seed. If a `RandomState` or
        `Generator` instance, the instance is used.
    figsize : (float,float), optional
        Figure size. If `None` it will be defined automatically.
    fill_band : bool, default True
        If True it fills in between to mark the area inside the confidence interval. Otherwise,
        plot the border lines.
    plot_kwargs : dict, optional
        Additional kwargs passed to :func:`mpl:matplotlib.pyplot.step` or
        :meth:`bokeh.plotting.figure.step`
    fill_kwargs : dict, optional
        Additional kwargs passed to :func:`mpl:matplotlib.pyplot.fill_between` or
        :meth:`bokeh:bokeh.plotting.Figure.varea`
    plot_outline_kwargs : dict, optional
        Additional kwargs passed to :meth:`mpl:matplotlib.axes.Axes.plot` or
        :meth:`bokeh:bokeh.plotting.Figure.line`
    ax :axes, optional
        Matplotlib axes or bokeh figures.
    show : bool, optional
        Call backend show function.
    backend : {"matplotlib", "bokeh"}, default "matplotlib"
        Select plotting backend.
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.
    confidence_bands : bool, default None
        deprecated: please see `band_kind`.
    pointwise : bool, default False
        deprecated: please see `band_kind`.
    fpr : float, optional
        deprecated: please see `band_prob`.

    Returns
    -------
    axes : matplotlib_axes or bokeh_figure

    References
    ----------
    .. [1] Säilynoja, T., Bürkner, P.C. and Vehtari, A. (2022). Graphical Test for
        Discrete Uniformity and its Applications in Goodness of Fit Evaluation and
        Multiple Sample Comparison. Statistics and Computing, 32(32).

    Examples
    --------
    Plot ecdf plot for a given sample

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> from scipy.stats import uniform, binom, norm

        >>> sample = norm(0,1).rvs(1000)
        >>> az.plot_ecdf(sample)

    Plot ecdf plot with confidence bands for comparing a given sample w.r.t a given distribution

    .. plot::
        :context: close-figs

        >>> distribution = norm(0,1)
        >>> az.plot_ecdf(sample, cdf = distribution.cdf, confidence_bands = True)

    Plot ecdf-difference plot with confidence bands for comparing a given sample
    w.r.t a given distribution

    .. plot::
        :context: close-figs

        >>> az.plot_ecdf(sample, cdf = distribution.cdf,
        >>>              confidence_bands = True, difference = True)

    Plot ecdf plot with confidence bands for PIT of sample for comparing a given sample
    w.r.t a given distribution

    .. plot::
        :context: close-figs

        >>> az.plot_ecdf(sample, cdf = distribution.cdf,
        >>>              confidence_bands = True, pit = True)

    Plot ecdf-difference plot with confidence bands for PIT of sample for comparing a given
    sample w.r.t a given distribution

    .. plot::
        :context: close-figs

        >>> az.plot_ecdf(sample, cdf = distribution.cdf,
        >>>              confidence_bands = True, difference = True, pit = True)

    You could also plot the above w.r.t another sample rather than a given distribution.
    For eg: Plot ecdf-difference plot with confidence bands for PIT of sample for
    comparing a given sample w.r.t a given sample

    .. plot::
        :context: close-figs

        >>> sample2 = norm(0,1).rvs(5000)
        >>> az.plot_ecdf(sample, sample2, confidence_bands = True, difference = True, pit = True)

    """
    if confidence_bands:
        warnings.warn(
            "confidence_bands keyword will be deprecated in a future release. Use `band_kind`.",
            FutureWarning,
        )

        if band_kind is not None:
            raise ValueError("Cannot specify both `confidence_bands` and `band_kind`")

        if pointwise:
            warnings.warn(
                "pointwise keyword will be deprecated in a future release. Use `band_kind='pointwise'`",
                FutureWarning,
            )
            band_kind = "pointwise"
        else:
            band_kind = "simulated"

    if fpr is not None:
        warnings.warn(
            "fpr keyword will be deprecated in a future release. Use `band_prob=1-fpr` "
            "or set rcParam `plot.band_prob` to `1-fpr`",
            FutureWarning,
        )
        if band_prob is not None:
            raise ValueError("Cannot specify both `fpr` and `band_prob`")
        band_prob = 1 - fpr

    if band_prob is None:
        band_prob = rcParams["plot.band_prob"]

    if values2 is None and cdf is None and band_kind is not None:
        raise ValueError("For confidence bands you need to specify values2 or the cdf")

    if cdf is not None and values2 is not None:
        raise ValueError("To compare sample you need either cdf or values2 and not both")

    if values2 is None and cdf is None and pit is True:
        raise ValueError("For PIT specify either cdf or values2")

    if values2 is None and cdf is None and difference is True:
        raise ValueError("For ECDF difference plot need either cdf or values2")

    if values2 is not None:
        values2 = np.ravel(values2)
        values2.sort()

    values = np.ravel(values)
    values.sort()

    if pit:
        eval_points = np.linspace(1 / npoints, 1, npoints)
        if cdf:
            sample = cdf(values)
        else:
            sample = compute_ecdf(values2, values) / len(values2)
        cdf_at_eval_points = eval_points
        rvs = uniform(0, 1).rvs
    else:
        eval_points = np.linspace(values[0], values[-1], npoints)
        sample = values
        if difference or band_kind is not None:
            if cdf:
                cdf_at_eval_points = cdf(eval_points)
            else:
                cdf_at_eval_points = compute_ecdf(values2, eval_points)
        else:
            cdf_at_eval_points = np.zeros_like(eval_points)

    x_coord, y_coord = _get_ecdf_points(sample, eval_points, difference)

    if difference:
        y_coord -= cdf_at_eval_points

    if band_kind is not None:
        ndraws = len(values)
        x_bands = eval_points
        lower, higher = ecdf_confidence_band(
            ndraws,
            eval_points,
            cdf_at_eval_points,
            method=band_kind,
            prob=band_prob,
            num_trials=num_trials,
            rvs=rvs,
            random_state=random_state,
        )

        if difference:
            lower -= cdf_at_eval_points
            higher -= cdf_at_eval_points
    else:
        x_bands, lower, higher = None, None, None

    ecdf_plot_args = dict(
        x_coord=x_coord,
        y_coord=y_coord,
        x_bands=x_bands,
        lower=lower,
        higher=higher,
        figsize=figsize,
        fill_band=fill_band,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        plot_outline_kwargs=plot_outline_kwargs,
        ax=ax,
        show=show,
        backend_kwargs=backend_kwargs,
        **kwargs,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_ecdf", "ecdfplot", backend)
    ax = plot(**ecdf_plot_args)

    return ax
