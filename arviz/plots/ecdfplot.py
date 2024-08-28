"""Plot ecdf or ecdf-difference plot with confidence bands."""

import warnings

import numpy as np
from scipy.stats import uniform

try:
    from scipy.stats import ecdf as scipy_ecdf
except ImportError:
    scipy_ecdf = None

from ..rcparams import rcParams
from ..stats.ecdf_utils import ecdf_confidence_band, _get_ecdf_points
from ..utils import BehaviourChangeWarning
from .plot_utils import get_plotting_function


def plot_ecdf(
    values,
    values2=None,
    eval_points=None,
    cdf=None,
    difference=False,
    confidence_bands=False,
    ci_prob=None,
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
    npoints=100,
    pointwise=False,
    fpr=None,
    pit=False,
    **kwargs,
):
    r"""Plot ECDF or ECDF-Difference Plot with Confidence bands.

    Plots of the empirical cumulative distribution function (ECDF) of an array. Optionally, A `cdf`
    argument representing a reference CDF may be provided for comparison using a difference ECDF
    plot and/or confidence bands.

    Alternatively, the PIT for a single dataset may be visualized.

    Notes
    -----
    This plot computes the confidence bands with the simulated based algorithm presented in [1]_.

    Parameters
    ----------
    values : array-like
        Values to plot from an unknown continuous or discrete distribution.
    values2 : array-like, optional
        values to compare to the original sample.

        .. deprecated:: 0.18.0
           Instead use ``cdf=scipy.stats.ecdf(values2).cdf.evaluate``.
    cdf : callable, optional
        Cumulative distribution function of the distribution to compare the original sample.
        The function must take as input a numpy array of draws from the distribution.
    difference : bool, default False
        If True then plot ECDF-difference plot otherwise ECDF plot.
    confidence_bands : str or bool

        - False: No confidence bands are plotted (default).
        - True: Plot bands computed with the default algorithm (subject to change)
        - "pointwise": Compute the pointwise (i.e. marginal) confidence band.
        - "optimized": Use optimization to estimate a simultaneous confidence band.
        - "simulated": Use Monte Carlo simulation to estimate a simultaneous confidence
          band.

        For simultaneous confidence bands to be correctly calibrated, provide `eval_points` that
        are not dependent on the `values`.
    ci_prob : float, default 0.94
        The probability that the true ECDF lies within the confidence band. If `confidence_bands`
        is "pointwise", this is the marginal probability instead of the joint probability.
    eval_points : array-like, optional
        The points at which to evaluate the ECDF. If None, `npoints` uniformly spaced points
        between the data bounds will be used.
    rvs: callable, optional
        A function that takes an integer `ndraws` and optionally the object passed to
        `random_state` and returns an array of `ndraws` samples from the same distribution
        as the original dataset. Required if `method` is "simulated" and variable is discrete.
    random_state : int, numpy.random.Generator or numpy.random.RandomState, optional
    num_trials : int, default 500
        The number of random ECDFs to generate for constructing simultaneous confidence bands
        (if `confidence_bands` is "simulated").
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
    npoints : int, default 100
        The number of evaluation points for the ecdf or ecdf-difference plots, if `eval_points` is
        not provided or `pit` is `True`.

        .. deprecated:: 0.18.0
           Instead specify ``eval_points=np.linspace(np.min(values), np.max(values), npoints)``
           unless `pit` is `True`.
    pointwise : bool, default False

        .. deprecated:: 0.18.0
           Instead use `confidence_bands="pointwise"`.
    fpr : float, optional

        .. deprecated:: 0.18.0
           Instead use `ci_prob=1-fpr`.
    pit : bool, default False
        If True plots the ECDF or ECDF-diff of PIT of sample.

        .. deprecated:: 0.18.0
           See below example instead.

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
    In a future release, the default behaviour of ``plot_ecdf`` will change.
    To maintain the original behaviour you should do:

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> import numpy as np
        >>> from scipy.stats import uniform, norm
        >>>
        >>> sample = norm(0,1).rvs(1000)
        >>> npoints = 100
        >>> az.plot_ecdf(sample, eval_points=np.linspace(sample.min(), sample.max(), npoints))

    However, seeing this warning isn't an indicator of anything being wrong,
    if you are happy to get different behaviour as ArviZ improves and adds
    new algorithms you can ignore it like so:

    .. plot::
        :context: close-figs

        >>> import warnings
        >>> warnings.filterwarnings("ignore", category=az.utils.BehaviourChangeWarning)

    Plot an ECDF plot for a given sample evaluated at the sample points. This will become
    the new behaviour when `eval_points` is not provided:

    .. plot::
        :context: close-figs

        >>> az.plot_ecdf(sample, eval_points=np.unique(sample))

    Plot an ECDF plot with confidence bands for comparing a given sample to a given distribution.
    We manually specify evaluation points independent of the values so that the confidence bands
    are correctly calibrated.

    .. plot::
        :context: close-figs

        >>> distribution = norm(0,1)
        >>> eval_points = np.linspace(*distribution.ppf([0.001, 0.999]), 100)
        >>> az.plot_ecdf(
        >>>     sample, eval_points=eval_points,
        >>>     cdf=distribution.cdf, confidence_bands=True
        >>> )

    Plot an ECDF-difference plot with confidence bands for comparing a given sample
    to a given distribution.

    .. plot::
        :context: close-figs

        >>> az.plot_ecdf(
        >>>     sample, cdf=distribution.cdf,
        >>>     confidence_bands=True, difference=True
        >>> )

    Plot an ECDF plot with confidence bands for the probability integral transform (PIT) of a
    continuous sample. If drawn from the reference distribution, the PIT values should be uniformly
    distributed.

    .. plot::
        :context: close-figs

        >>> pit_vals = distribution.cdf(sample)
        >>> uniform_dist = uniform(0, 1)
        >>> az.plot_ecdf(
        >>>     pit_vals, cdf=uniform_dist.cdf, confidence_bands=True,
        >>> )

    Plot an ECDF-difference plot of PIT values.

    .. plot::
        :context: close-figs

        >>> az.plot_ecdf(
        >>>     pit_vals, cdf = uniform_dist.cdf, confidence_bands = True,
        >>>     difference = True
        >>> )
    """
    if confidence_bands is True:
        if pointwise:
            warnings.warn(
                "`pointwise` has been deprecated. Use `confidence_bands='pointwise'` instead.",
                FutureWarning,
            )
            confidence_bands = "pointwise"
        else:
            confidence_bands = "auto"
        # if pointwise specified, confidence_bands must be a bool or 'pointwise'
    elif confidence_bands not in [False, "pointwise"] and pointwise:
        raise ValueError(
            f"Cannot specify both `confidence_bands='{confidence_bands}'` and `pointwise=True`"
        )

    if fpr is not None:
        warnings.warn(
            "`fpr` has been deprecated. Use `ci_prob=1-fpr` or set `rcParam['stats.ci_prob']` to"
            "`1-fpr`.",
            FutureWarning,
        )
        if ci_prob is not None:
            raise ValueError("Cannot specify both `fpr` and `ci_prob`")
        ci_prob = 1 - fpr

    if ci_prob is None:
        ci_prob = rcParams["stats.ci_prob"]

    if values2 is not None:
        if cdf is not None:
            raise ValueError("You cannot specify both `values2` and `cdf`")
        if scipy_ecdf is None:
            raise ValueError(
                "The `values2` argument is deprecated and `scipy.stats.ecdf` is not available. "
                "Please use `cdf` instead."
            )
        warnings.warn(
            "`values2` has been deprecated. Use `cdf=scipy.stats.ecdf(values2).cdf.evaluate` "
            "instead.",
            FutureWarning,
        )
        cdf = scipy_ecdf(np.ravel(values2)).cdf.evaluate

    if cdf is None:
        if confidence_bands:
            raise ValueError("For confidence bands you must specify cdf")
        if difference is True:
            raise ValueError("For ECDF difference plot you must specify cdf")
        if pit:
            raise ValueError("For PIT plot you must specify cdf")

    values = np.ravel(values)
    values.sort()

    if pit:
        warnings.warn(
            "`pit` has been deprecated. Specify `values=cdf(values)` instead.",
            FutureWarning,
        )
        values = cdf(values)
        cdf = uniform(0, 1).cdf
        rvs = uniform(0, 1).rvs
        eval_points = np.linspace(1 / npoints, 1, npoints)

    if eval_points is None:
        warnings.warn(
            "In future versions, if `eval_points` is not provided, then the ECDF will be evaluated"
            " at the unique values of the sample. To keep the current behavior, provide "
            "`eval_points` explicitly.",
            BehaviourChangeWarning,
        )
        if confidence_bands in ["optimized", "simulated"]:
            warnings.warn(
                "For simultaneous bands to be correctly calibrated, specify `eval_points` "
                "independent of the `values`"
            )
        eval_points = np.linspace(values[0], values[-1], npoints)
    else:
        eval_points = np.asarray(eval_points)

    if difference or confidence_bands:
        cdf_at_eval_points = cdf(eval_points)
    else:
        cdf_at_eval_points = np.zeros_like(eval_points)

    x_coord, y_coord = _get_ecdf_points(values, eval_points, difference)

    if difference:
        y_coord -= cdf_at_eval_points

    if confidence_bands:
        ndraws = len(values)
        if confidence_bands == "auto":
            if ndraws < 200 or num_trials >= 250 * np.sqrt(ndraws):
                confidence_bands = "optimized"
            else:
                confidence_bands = "simulated"
        x_bands = eval_points
        lower, higher = ecdf_confidence_band(
            ndraws,
            eval_points,
            cdf_at_eval_points,
            method=confidence_bands,
            prob=ci_prob,
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
