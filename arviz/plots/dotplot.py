import math
import numpy as np

from ..rcparams import rcParams
from .plot_utils import get_plotting_function


def plot_dots(
    values=None,
    binwidth=None,
    dotsize=1,
    stackratio=1,
    hdi_prob=None,
    rotated=False,
    dotcolor="grey",
    intervalcolor="red",
    markersize=None,
    figsize=None,
    linewidth=None,
    point_estimate="auto",
    quantiles=None,
    point_interval=None,
    plot_kwargs=None,
    interval_kwargs=None,
    backend=None,
    backend_kwargs=None,
    **kwargs
):

    """Plot distribution as dot plot.

    If quantiles is specified then it will plot the quantile dot plot.

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
    hdi_prob : float
        Valid only when point_interval is True. Plots HDI for chosen percentage of density. 
        Defaults to 0.94.
    rotated : bool
        Whether to rotate the dot plot by 90 degrees.
    quantiles : int
        Number of quantiles to plot, used for quantile dot plots
        Defaults to None.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    plot_kwargs : dict
        Keywords passed for customizing the dots.
    interval_kwargs : dict
        Keyword passed to the point interval
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.

    Returns
    -------
    axes : matplotlib axes
    
    """

    if values is None:
        raise ValueError("Please provide the values array for plotting")

    values = np.asarray(values)
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

    if quantiles:
        if quantiles >= values.shape[0]:
            quantiles = values.shape[0]
        else:
            qlist = np.arange(100 / (2 * quantiles), 100, 100 / (quantiles))
            values = np.percentile(values, qlist)
    else:
        quantiles = values.shape[0]

    if binwidth is None:
        binwidth = math.sqrt(((((values[-1] - values[0]) ** 2) / 2) / quantiles) / np.pi)

    dot_plot_args = dict(
        values=values,
        binwidth=binwidth,
        dotsize=dotsize,
        stackratio=stackratio,
        hdi_prob=hdi_prob,
        rotated=rotated,
        dotcolor=dotcolor,
        intervalcolor=intervalcolor,
        markersize=markersize,
        figsize=figsize,
        linewidth=linewidth,
        point_estimate=point_estimate,
        quantiles=quantiles,
        point_interval=point_interval,
        backend_kwargs=backend_kwargs,
        plot_kwargs=plot_kwargs,
        interval_kwargs=interval_kwargs,
        **kwargs
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_dots", "dotplot", backend)
    ax = plot(**dot_plot_args)

    return ax
