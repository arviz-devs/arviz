"""Plot distribution as dot plot or quantile dot plot."""

import numpy as np


from ..rcparams import rcParams
from .plot_utils import get_plotting_function


def plot_dot(
    values,
    binwidth=None,
    dotsize=1,
    stackratio=1,
    hdi_prob=None,
    rotated=False,
    dotcolor="C0",
    intervalcolor="C3",
    markersize=None,
    markercolor="C0",
    marker="o",
    figsize=None,
    linewidth=None,
    point_estimate="auto",
    nquantiles=50,
    quartiles=True,
    point_interval=False,
    ax=None,
    show=None,
    plot_kwargs=None,
    backend=None,
    backend_kwargs=None,
    **kwargs
):
    r"""Plot distribution as dot plot or quantile dot plot.

    This function uses the Wilkinson's Algorithm [1]_ to allot dots to bins.
    The quantile dot plots was inspired from [2]_.

    Parameters
    ----------
    values : array-like
        Values to plot from an unknown continuous or discrete distribution.
    binwidth : float, optional
        Width of the bin for drawing the dot plot.
    dotsize : float, default 1
        The size of the dots relative to the bin width. The default makes dots be
        just about as wide as the bin width.
    stackratio : float, default 1
        The distance between the center of the dots in the same stack relative to the bin height.
        The default makes dots in the same stack just touch each other.
    point_interval : bool, default False
        Plots the point interval. Uses ``hdi_prob`` to plot the HDI interval
    point_estimate : str, optional
        Plot point estimate per variable. Values should be ``mean``, ``median``, ``mode`` or None.
        Defaults to ``auto`` i.e. it falls back to default set in rcParams.
    dotcolor : string, optional
        The color of the dots. Should be a valid matplotlib color.
    intervalcolor : string, optional
        The color of the interval. Should be a valid matplotlib color.
    linewidth : int, default None
        Line width throughout. If None it will be autoscaled based on `figsize`.
    markersize : int, default None
        Markersize throughout. If None it will be autoscaled based on `figsize`.
    markercolor : string, optional
        The color of the marker when plot_interval is True. Should be a valid matplotlib color.
    marker : string, default "o"
        The shape of the marker. Valid for matplotlib backend.
    hdi_prob : float, optional
        Valid only when point_interval is True. Plots HDI for chosen percentage of density.
        Defaults to ``stats.ci_prob`` rcParam. See :ref:`this section <common_hdi_prob>`
        for usage examples.
    rotated : bool, default False
        Whether to rotate the dot plot by 90 degrees.
    nquantiles : int, default 50
        Number of quantiles to plot, used for quantile dot plots.
    quartiles : bool, default True
        If True then the quartile interval will be plotted with the HDI.
    figsize : (float,float), optional
        Figure size. If ``None`` it will be defined automatically.
    plot_kwargs : dict, optional
        Keywords passed for customizing the dots. Passed to :class:`mpl:matplotlib.patches.Circle`
        in matplotlib and :meth:`bokeh.plotting.figure.circle` in bokeh.
    backend :{"matplotlib", "bokeh"}, default "matplotlib"
        Select plotting backend.
    ax : axes, optional
        Matplotlib_axes or bokeh_figure.
    show : bool, optional
        Call backend show function.
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.

    Returns
    -------
    axes : matplotlib_axes or bokeh_figure

    See Also
    --------
    plot_dist : Plot distribution as histogram or kernel density estimates.

    References
    ----------
    .. [1] Leland Wilkinson (1999) Dot Plots, The American Statistician, 53:3, 276-281,
        DOI: 10.1080/00031305.1999.10474474
    .. [2] Matthew Kay, Tara Kola, Jessica R. Hullman,
        and Sean A. Munson. 2016. When (ish) is My Bus? User-centered Visualizations of Uncertainty
        in Everyday, Mobile Predictive Systems. DOI:https://doi.org/10.1145/2858036.2858558

    Examples
    --------
    Plot dot plot for a set of data points

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> import numpy as np
        >>> values = np.random.normal(0, 1, 500)
        >>> az.plot_dot(values)

    Manually adjust number of quantiles to plot

    .. plot::
        :context: close-figs

        >>> az.plot_dot(values, nquantiles=100)

    Add a point interval under the dot plot

    .. plot::
        :context: close-figs

        >>> az.plot_dot(values, point_interval=True)

    Rotate the dot plots by 90 degrees i.e swap x and y axis

    .. plot::
        :context: close-figs

        >>> az.plot_dot(values, point_interval=True, rotated=True)

    """
    if nquantiles == 0:
        raise ValueError("Number of quantiles should be greater than 0")

    if marker != "o" and backend == "bokeh":
        raise ValueError("marker argument is valid only for matplotlib backend")

    values = np.ravel(values)
    values = values[np.isfinite(values)]
    values.sort()

    if hdi_prob is None:
        hdi_prob = rcParams["stats.ci_prob"]
    elif not 1 >= hdi_prob > 0:
        raise ValueError("The value of hdi_prob should be in the interval (0, 1]")

    if point_estimate == "auto":
        point_estimate = rcParams["plot.point_estimate"]
    elif point_estimate not in {"mean", "median", "mode", None}:
        raise ValueError("The value of point_estimate must be either mean, median, mode or None.")

    if not isinstance(nquantiles, int):
        raise TypeError("nquantiles must be of integer type, refer to docs for further details")

    dot_plot_args = dict(
        values=values,
        binwidth=binwidth,
        dotsize=dotsize,
        stackratio=stackratio,
        hdi_prob=hdi_prob,
        quartiles=quartiles,
        rotated=rotated,
        dotcolor=dotcolor,
        intervalcolor=intervalcolor,
        markersize=markersize,
        markercolor=markercolor,
        marker=marker,
        figsize=figsize,
        linewidth=linewidth,
        point_estimate=point_estimate,
        nquantiles=nquantiles,
        point_interval=point_interval,
        ax=ax,
        show=show,
        backend_kwargs=backend_kwargs,
        plot_kwargs=plot_kwargs,
        **kwargs
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_dot", "dotplot", backend)
    ax = plot(**dot_plot_args)

    return ax


def wilkinson_algorithm(values, binwidth):
    """Wilkinson's algorithm to distribute dots into horizontal stacks."""
    ndots = len(values)
    count = 0
    stack_locs, stack_counts = [], []

    while count < ndots:
        stack_first_dot = values[count]
        num_dots_stack = 0
        while values[count] < (binwidth + stack_first_dot):
            num_dots_stack += 1
            count += 1
            if count == ndots:
                break
        stack_locs.append((stack_first_dot + values[count - 1]) / 2)
        stack_counts.append(num_dots_stack)

    return stack_locs, stack_counts


def layout_stacks(stack_locs, stack_counts, binwidth, stackratio, rotated):
    """Use count and location of stacks to get coordinates of dots."""
    dotheight = stackratio * binwidth
    binradius = binwidth / 2

    x = np.repeat(stack_locs, stack_counts)
    y = np.hstack([dotheight * np.arange(count) + binradius for count in stack_counts])
    if rotated:
        x, y = y, x

    return x, y
