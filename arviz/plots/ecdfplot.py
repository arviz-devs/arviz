"""Plot ecdf or ecdf-difference plot with confidence bands."""
import numpy as np
from scipy.stats import uniform, binom

from arviz.rcparams import rcParams
from arviz.plots.plot_utils import get_plotting_function


def plot_ecdf(
    values,
    values2=None,
    distribution=None,
    difference=False,
    pit=False,
    confidence_bands=True,
    granularity=100,
    num_trials=500,
    alpha=0.05,
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
    """Plot ECDF or ECDF-Difference Plot with Confidence bands.

    This plot uses the simulated based algorithm presented in the paper "Graphical Test for
    Discrete Uniformity and its Applications in Goodness of Fit Evaluation and
    Multiple Sample Comparison" [1]_.

    Parameters
    ----------
    values : array-like
        Values to plot from an unknown continuous or discrete distribution
    values2 : array-like, optional
        Values to compare to the original sample
    distribution : function, optional
        Cumulative distribution function of the distribution to compare the original sample to
    difference : bool, optional
        If true then plot ECDF-difference plot otherwise ECDF plot
    pit : bool, optional
        If True plots the ECDF or ECDF-diff of PIT of sample
    confidence_bands : bool, optional
        If True plots the simultaneous confidence bands with 1 - alpha confidence level
    granularity : int, optional, Defaults 100
        This denotes the granularity size of our plot
        i.e the number of evaluation points for our ecdf or ecdf-difference plot
    num_trials : int, optional, Defaults 500
        The number of random ECDFs to generate to construct simultaneous confidence bands
    alpha : float, optional, Defaults 0.05
        The type I error rate s.t 1 - alpha denotes the confidence level of bands
    figsize : tuple, optional
        Figure size. If None it will be defined automatically.
    fill_band : bool, optional
        Use fill_between to mark the area inside the credible interval.
        Otherwise, plot the border lines.
    plot_kwargs : dict, optional
        Additional kwargs passed to :func:`mpl:matplotlib.pyplot.step` or
        :meth:`bokeh:bokeh.plotting.Figure.step`
    fill_kwargs : dict, optional
        Additional kwargs passed to :func:`mpl:matplotlib.pyplot.fill_between` or
        :meth:`bokeh:bokeh.plotting.Figure.varea`
    plot_outline_kwargs : dict, optional
        Additional kwargs passed to :meth:`mpl:matplotlib.axes.Axes.plot` or
        :meth:`bokeh:bokeh.plotting.Figure.line`
    ax : axes, optional
        Matplotlib axes or bokeh figures.
    show : bool, optional
        Call backend show function.
    backend : str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`mpl:matplotlib.pyplot.subplots` or
        :meth:`bokeh:bokeh.plotting.figure`.

    Returns
    -------
    axes : matplotlib axes or bokeh figures

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
        >>> az.plot_ecdf(sample, distribution = distribution.cdf, confidence_bands = True)

    Plot ecdf-difference plot with confidence bands for comparing a given sample
    w.r.t a given distribution

    .. plot::
        :context: close-figs

        >>> az.plot_ecdf(sample, distribution = distribution.cdf,
            confidence_bands = True, difference = True)

    Plot ecdf plot with confidence bands for PIT of sample for comparing a given sample
    w.r.t a given distribution

    .. plot::
        :context: close-figs

        >>> az.plot_ecdf(sample, distribution = distribution.cdf,
            confidence_bands = True, pit = True)

    Plot ecdf-difference plot with confidence bands for PIT of sample for comparing a given
    sample w.r.t a given distribution

    .. plot::
        :context: close-figs

        >>> az.plot_ecdf(sample, distribution = distribution.cdf,
            confidence_bands = True, difference = True, pit = True)

    You could also plot the above w.r.t another sample rather than a given distribution.
    For eg: Plot ecdf-difference plot with confidence bands for PIT of sample for
    comparing a given sample w.r.t a given sample

    .. plot::
        :context: close-figs

        >>> sample2 = norm(0,1).rvs(5000)
        >>> az.plot_ecdf(sample, sample2, confidence_bands = True, difference = True, pit = True)

    """
    if values2 is None and distribution is None and confidence_bands is True:
        raise ValueError("For confidence bands you need to specify values2 or the distribution")

    if distribution is not None and values2 is not None:
        raise ValueError("To compare sample you need either distribution or values2 and not both")

    if values2 is None and distribution is None and pit is True:
        raise ValueError("For PIT specify either distribution or values2")

    if values2 is None and distribution is None and difference is True:
        raise ValueError("For ECDF difference plot need either distribution or values2")

    if values2 is not None:
        values2 = np.ravel(values2)
        values2.sort()

    values = np.ravel(values)
    values.sort()

    n = len(values)  # number of samples
    ## This block computes gamma and uses it to get the upper and lower confidence bands
    ## Here we check if we want confidence bands or not
    if confidence_bands:
        ## If plotting PIT then we find the PIT values of sample
        if pit:
            x = np.linspace(1 / granularity, 1, granularity)
            z = x
            ## Finding PIT for our sample
            probs = (
                distribution(values)
                if distribution
                else compute_ecdf(values2, values) / len(values2)
            )
        else:
            x = np.linspace(values[0], values[-1], granularity)
            z = distribution(x) if distribution else compute_ecdf(values2, x)
            probs = values

        ## Computing gamma
        gamma = compute_gamma(n, z, granularity, num_trials, alpha)
        ## Using gamma to get the confidence intervals
        lower, higher = get_lims(gamma, n, z)

        ## This block is for whether to plot ECDF or ECDF-difference
        if not difference:
            ## We store the coordinates of our ecdf in x_coord, y_coord
            x_coord, y_coord = np.empty(len(x) + 1), np.empty(len(x) + 1)
            ## pseudo point at the start of ecdf plot so that it touches x-axis
            x_coord[0], y_coord[0] = x[0], 0
            for i, x_i in enumerate(x):
                f_x_i = compute_ecdf(probs, x_i)
                x_coord[i + 1], y_coord[i + 1] = x_i, f_x_i
        else:
            ## Here we subtract the ecdf value as here we are plotting the ECDF-difference
            x_coord, y_coord = np.empty(len(x)), np.empty(len(x))
            for i, x_i in enumerate(x):
                f_x_i = compute_ecdf(probs, x_i) - (
                    x_i
                    if pit
                    else distribution(x_i)
                    if distribution
                    else compute_ecdf(values2, x_i)
                )
                x_coord[i], y_coord[i] = x_i, f_x_i

            ## Similarly we subtract from the upper and lower bounds
            if pit:
                lower = lower - x
                higher = higher - x
            else:
                lower = lower - (distribution(x) if distribution else compute_ecdf(values2, x))
                higher = higher - (distribution(x) if distribution else compute_ecdf(values2, x))

    else:
        if pit:
            x = np.linspace(1 / granularity, 1, granularity)
            probs = distribution(values)
        else:
            x = np.linspace(values[0], values[-1], granularity)
            probs = values

        lower, higher = None, None
        ## This block is for whether to plot ECDF or ECDF-difference
        if not difference:
            x_coord, y_coord = np.empty(len(x) + 1), np.empty(len(x) + 1)
            ## pseudo point at the start of ecdf plot sp that it touches x-axis
            x_coord[0], y_coord[0] = x[0], 0
            for i, x_i in enumerate(x):
                f_x_i = compute_ecdf(probs, x_i)
                x_coord[i + 1], y_coord[i + 1] = x_i, f_x_i
        else:
            x_coord, y_coord = np.empty(len(x)), np.empty(len(x))
            ## Here we subtract the ecdf value as here we are plotting the ECDF-difference
            for i, x_i in enumerate(x):
                f_x_i = compute_ecdf(probs, x_i) - (
                    x_i
                    if pit
                    else distribution(x_i)
                    if distribution
                    else compute_ecdf(values2, x_i)
                )
                x_coord[i], y_coord[i] = x_i, f_x_i

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
    if not isinstance(z, np.ndarray):
        ## if z is just an instance then use Binary search
        left, right = 0, len(sample) - 1
        while left <= right:
            mid = int((left + right) / 2)
            if sample[mid] > z:
                right = mid - 1
            else:
                left = mid + 1
        return left / len(sample)
    else:
        ## if z is a list then follow this approach
        f_z = np.empty(len(z))
        u_idx = 0
        for i, z_i in enumerate(z):
            while u_idx < len(sample) and sample[u_idx] < z_i:
                u_idx += 1
            f_z[i] = u_idx + 1

        return f_z / len(sample)


def compute_gamma(n, z, granularity=None, num_trials=1000, alpha=0.05):
    """Compute gamma for confidence interval calculation.

    This function simulates an adjusted value of gamma to account for multiplicity
    when forming an 1-alpha level confidence envelope for the ECDF of a sample.
    """
    if granularity is None:
        granularity = n
    gamma = []
    for _ in range(num_trials):
        unif_samples = uniform.rvs(0, 1, n)
        unif_samples = np.sort(unif_samples)
        gamma_m = 1000
        ## Can compute ecdf for all the z together or one at a time.
        f_z = compute_ecdf(unif_samples, z)
        for i in range(granularity):
            curr = min(binom.cdf(n * f_z[i], n, z[i]), 1 - binom.cdf(n * f_z[i] - 1, n, z[i]))
            gamma_m = min(2 * curr, gamma_m)
        gamma.append(gamma_m)
    return np.quantile(gamma, alpha)


def get_lims(gamma, n, z):
    """Compute the simultaneous 1 - alpha level confidence bands."""
    lower = binom.ppf(gamma / 2, n, z)
    upper = binom.ppf(1 - gamma / 2, n, z)
    return lower / n, upper / n
