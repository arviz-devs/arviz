"""Plot ecdf or ecdf-difference plot with confidence bands."""
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

    ## This block computes gamma and uses it to get the upper and lower confidence bands
    ## Here we check if we want confidence bands or not
    if confidence_bands:
        ## If plotting PIT then we find the PIT values of sample.
        ## Basically here we generate the evaluation points(x) and find the PIT values.
        ## z is the evaluation point for our uniform distribution in compute_gamma()
        if pit:
            x = np.linspace(1 / npoints, 1, npoints)
            z = x
            ## Finding PIT for our sample
            probs = cdf(values) if cdf else compute_ecdf(values2, values) / len(values2)
        else:
            ## If not PIT use sample for plots and for evaluation points(x) use equally spaced
            ## points between minimum and maximum of sample
            ## For z we have used cdf(x)
            x = np.linspace(values[0], values[-1], npoints)
            z = cdf(x) if cdf else compute_ecdf(values2, x)
            probs = values

        n = len(values)  # number of samples
        ## Computing gamma
        gamma = fpr if pointwise else compute_gamma(n, z, npoints, num_trials, fpr)
        ## Using gamma to get the confidence intervals
        lower, higher = get_lims(gamma, n, z)

        ## This block is for whether to plot ECDF or ECDF-difference
        if not difference:
            ## We store the coordinates of our ecdf in x_coord, y_coord
            x_coord, y_coord = get_ecdf_points(x, probs, difference)
        else:
            ## Here we subtract the ecdf value as here we are plotting the ECDF-difference
            x_coord, y_coord = get_ecdf_points(x, probs, difference)
            for i, x_i in enumerate(x):
                y_coord[i] = y_coord[i] - (
                    x_i if pit else cdf(x_i) if cdf else compute_ecdf(values2, x_i)
                )

            ## Similarly we subtract from the upper and lower bounds
            if pit:
                lower = lower - x
                higher = higher - x
            else:
                lower = lower - (cdf(x) if cdf else compute_ecdf(values2, x))
                higher = higher - (cdf(x) if cdf else compute_ecdf(values2, x))

    else:
        if pit:
            x = np.linspace(1 / npoints, 1, npoints)
            probs = cdf(values)
        else:
            x = np.linspace(values[0], values[-1], npoints)
            probs = values

        lower, higher = None, None
        ## This block is for whether to plot ECDF or ECDF-difference
        if not difference:
            x_coord, y_coord = get_ecdf_points(x, probs, difference)
        else:
            ## Here we subtract the ecdf value as here we are plotting the ECDF-difference
            x_coord, y_coord = get_ecdf_points(x, probs, difference)
            for i, x_i in enumerate(x):
                y_coord[i] = y_coord[i] - (
                    x_i if pit else cdf(x_i) if cdf else compute_ecdf(values2, x_i)
                )

    ecdf_plot_args = dict(
        x_coord=x_coord,
        y_coord=y_coord,
        x_bands=x,
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


def compute_ecdf(sample, z):
    """Compute ECDF.

    This function computes the ecdf value at the evaluation point
        or a sorted set of evaluation points.
    """
    return np.searchsorted(sample, z, side="right") / len(sample)


def get_ecdf_points(x, probs, difference):
    """Compute the coordinates for the ecdf points using compute_ecdf."""
    y = compute_ecdf(probs, x)

    if not difference:
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, 0)
    return x, y


def compute_gamma(n, z, npoints=None, num_trials=1000, fpr=0.05):
    """Compute gamma for confidence interval calculation.

    This function simulates an adjusted value of gamma to account for multiplicity
    when forming an 1-fpr level confidence envelope for the ECDF of a sample.
    """
    if npoints is None:
        npoints = n
    gamma = []
    for _ in range(num_trials):
        unif_samples = uniform.rvs(0, 1, n)
        unif_samples = np.sort(unif_samples)
        gamma_m = 1000
        ## Can compute ecdf for all the z together or one at a time.
        f_z = compute_ecdf(unif_samples, z)
        f_z = compute_ecdf(unif_samples, z)
        gamma_m = 2 * min(
            np.amin(binom.cdf(n * f_z, n, z)), np.amin(1 - binom.cdf(n * f_z - 1, n, z))
        )
        gamma.append(gamma_m)
    return np.quantile(gamma, fpr)


def get_lims(gamma, n, z):
    """Compute the simultaneous 1 - fpr level confidence bands."""
    lower = binom.ppf(gamma / 2, n, z)
    upper = binom.ppf(1 - gamma / 2, n, z)
    return lower / n, upper / n
