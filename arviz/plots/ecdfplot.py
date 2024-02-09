"""Plot ecdf or ecdf-difference plot with confidence bands."""
from typing import Tuple

import numpy as np
from scipy.stats import uniform, binom

from ..rcparams import rcParams
from .plot_utils import get_plotting_function


def plot_ecdf(
    values,
    values2=None,
    cdf=None,
    difference=False,
    pit=False,
    confidence_bands=None,
    pointwise=False,
    npoints=100,
    num_trials=500,
    fpr=0.05,
    figsize=None,
    fill_band=True,
    plot_kwargs=None,
    fill_kwargs=None,
    plot_outline_kwargs=None,
    ax=None,
    show=None,
    backend=None,
    backend_kwargs=None,
    **kwargs
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
    difference : bool, default False
        If True then plot ECDF-difference plot otherwise ECDF plot.
    pit : bool, default False
        If True plots the ECDF or ECDF-diff of PIT of sample.
    confidence_bands : bool, default None
        If True plots the simultaneous or pointwise confidence bands with `1 - fpr`
        confidence level.
    pointwise : bool, default False
        If True plots pointwise confidence bands otherwise simultaneous bands.
    npoints : int, default 100
        This denotes the granularity size of our plot i.e the number of evaluation points
        for the ecdf or ecdf-difference plots.
    num_trials : int, default 500
        The number of random ECDFs to generate for constructing simultaneous confidence bands.
    fpr : float, default 0.05
        The type I error rate s.t `1 - fpr` denotes the confidence level of bands.
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

    Returns
    -------
    axes : matplotlib_axes or bokeh_figure

    References
    ----------
    .. [1] Säilynoja, T., Bürkner, P.C. and Vehtari, A., 2021. Graphical Test for
        Discrete Uniformity and its Applications in Goodness of Fit Evaluation and
        Multiple Sample Comparison. arXiv preprint arXiv:2103.10522.

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
    if confidence_bands is None:
        confidence_bands = (values2 is not None) or (cdf is not None)

    if values2 is None and cdf is None and confidence_bands is True:
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
        sample = cdf(values) if cdf else compute_ecdf(values2, values) / len(values2)
        cdf_at_eval_points = eval_points
    else:
        eval_points = np.linspace(values[0], values[-1], npoints)
        sample = values
        if confidence_bands or difference:
            cdf_at_eval_points = cdf(eval_points) if cdf else compute_ecdf(values2, eval_points)
        else:
            cdf_at_eval_points = np.zeros_like(eval_points)

    x_coord, y_coord = get_ecdf_points(sample, eval_points, difference)

    if difference:
        y_coord -= cdf_at_eval_points

    if confidence_bands:
        ndraws = len(values)  # number of samples
        if pointwise:
            prob_pointwise = 1 - fpr
        else:
            prob_pointwise = simulate_simultaneous_band_probability(
                ndraws, eval_points, cdf_at_eval_points, num_trials=num_trials, prob_target=1 - fpr
            )
        lower, higher = get_pointwise_confidence_band(prob_pointwise, ndraws, cdf_at_eval_points)

        if difference:
            lower -= cdf_at_eval_points
            higher -= cdf_at_eval_points
    else:
        lower, higher = None, None

    ecdf_plot_args = dict(
        x_coord=x_coord,
        y_coord=y_coord,
        x_bands=eval_points,
        lower=lower,
        higher=higher,
        confidence_bands=confidence_bands,
        figsize=figsize,
        fill_band=fill_band,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        plot_outline_kwargs=plot_outline_kwargs,
        ax=ax,
        show=show,
        backend_kwargs=backend_kwargs,
        **kwargs
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_ecdf", "ecdfplot", backend)
    ax = plot(**ecdf_plot_args)

    return ax


def compute_ecdf(sample, eval_points):
    """Compute ECDF.

    This function computes the ecdf value at the evaluation point
        or a sorted set of evaluation points.
    """
    return np.searchsorted(sample, eval_points, side="right") / len(sample)


def get_ecdf_points(sample, eval_points, difference):
    """Compute the coordinates for the ecdf points using compute_ecdf."""
    x = eval_points
    y = compute_ecdf(sample, eval_points)

    if not difference:
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, 0)
    return x, y


def simulate_simultaneous_band_probability(
    ndraws, eval_points, cdf_at_eval_points, num_trials=1000, prob_target=0.95
):
    """Estimate probability for simultaneous confidence band using simulation.

    This function simulates the pointwise probability needed to construct pointwise
    confidence bands that form a `prob_target`-level confidence envelope for the ECDF
    of a sample.
    """
    probs = []
    for _ in range(num_trials):
        unif_samples = uniform.rvs(0, 1, ndraws)
        unif_samples.sort()
        ecdf_at_eval_points = compute_ecdf(unif_samples, cdf_at_eval_points)
        prob = fit_pointwise_band_probability(ndraws, ecdf_at_eval_points, cdf_at_eval_points)
        probs.append(prob)
    return np.quantile(probs, prob_target)


def fit_pointwise_band_probability(
    ndraws: int,
    ecdf_at_eval_points: np.ndarray,
    cdf_at_eval_points: np.ndarray,
) -> float:
    """Compute the smallest marginal probability of a pointwise confidence band that
    contains the ECDF."""
    ecdf_scaled = (ndraws * ecdf_at_eval_points).astype(int)
    prob_lower_tail = np.amin(binom.cdf(ecdf_scaled, ndraws, cdf_at_eval_points))
    prob_upper_tail = np.amin(binom.sf(ecdf_scaled - 1, ndraws, cdf_at_eval_points))
    prob_pointwise = 1 - 2 * min(prob_lower_tail, prob_upper_tail)
    return prob_pointwise


def get_pointwise_confidence_band(
    prob: float, num_draws: int, cdf_at_eval_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the `prob`-level pointwise confidence band."""
    prob_lower_tail = (1 - prob) / 2
    prob_upper_tail = 1 - prob_lower_tail
    prob_tails = np.array([[prob_lower_tail, prob_upper_tail]]).T
    prob_lower, prob_upper = binom.ppf(prob_tails, num_draws, cdf_at_eval_points) / num_draws
    return prob_lower, prob_upper
