import numpy as np

from ..rcparams import rcParams
from .plot_utils import get_plotting_function


def plot_dot(
    values=None,
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
    point_interval=None,
    ax=None,
    show=None,
    plot_kwargs=None,
    backend=None,
    backend_kwargs=None,
    **kwargs
):

    """Plot distribution as dot plot or quantile dot plot.

    This function uses the Wilkinson's Algorithm
    (Leland Wilkinson (1999) Dot Plots, The American Statistician, 53:3, 276-281,
    DOI: 10.1080/00031305.1999.10474474)to allot dots to bins.
    The quantile dot plots was inspired from the paper(Matthew Kay, Tara Kola, Jessica R. Hullman,
    and Sean A. Munson. 2016. When (ish) is My Bus? User-centered Visualizations of Uncertainty in
    Everyday, Mobile Predictive Systems. DOI:https://doi.org/10.1145/2858036.2858558).

    Parameters
    ----------
    values : array-like
        Values to plot
    binwidth : float
        Width of the bin for drawing the dot plot.
    dotsize : float
        The size of the dots relative to the bin width. The default, 1, makes dots be
        just about as wide as the bin width.
    stackratio : float
        The distance between the center of the dots in the same stack relative to the bin height.
        The default, 1, makes dots in the same stack just touch each other.
    point_interval : bool
        Plots the point interval. Uses hdi_prob to plot the HDI interval
    point_estimate : Optional[str]
        Plot point estimate per variable. Values should be ‘mean’, ‘median’, ‘mode’ or None.
        Defaults to ‘auto’ i.e. it falls back to default set in rcParams.
    dotcolor : string
        The color of the dots
    intervalcolor : string
        The color of the interval
    linewidth : int
        Line width throughout. If None it will be autoscaled based on figsize.
    markersize : int
        Markersize throughout. If None it will be autoscaled based on figsize.
    markercolor: string
        The color of the marker when plot_interval is True
    marker: string
        The shape of the marker. Valid for matplotlib backend
        Defaults to "o"
    hdi_prob : float
        Valid only when point_interval is True. Plots HDI for chosen percentage of density.
        Defaults to 0.94.
    rotated : bool
        Whether to rotate the dot plot by 90 degrees.
    nquantiles : int
        Number of quantiles to plot, used for quantile dot plots
        Defaults to 50.
    quartiles : bool
        If True then the quartile interval will be plotted with the HDI.
        Defaults to True.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    plot_kwargs : dict
        Keywords passed for customizing the dots.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    ax : axes, optional
        Matplotlib axes or bokeh figures.
    show: bool, optional
        Call backend show function.
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.

    Returns
    -------
    axes : matplotlib axes

    """

    if values is None:
        raise ValueError("Please provide the values array for plotting")

    if nquantiles == 0:
        raise ValueError("Number of quantiles should be greater than 0")

    if marker != "o" and backend == "bokeh":
        raise ValueError("marker argument is valid only for matplotlib backend")

    values = np.sort(values)

    if hdi_prob is None:
        hdi_prob = rcParams["stats.hdi_prob"]
    else:
        if not 1 >= hdi_prob > 0:
            raise ValueError("The value of hdi_prob should be in the interval (0, 1]")

    if point_estimate == "auto":
        point_estimate = rcParams["plot.point_estimate"]
    elif point_estimate not in {"mean", "median", "mode", None}:
        raise ValueError("The value of point_estimate must be either mean, median, mode or None.")

    if isinstance(nquantiles, (bool, str)):
        raise ValueError("quantiles must be of integer type, refer to docs for further details")

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
