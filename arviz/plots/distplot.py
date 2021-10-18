# pylint: disable=unexpected-keyword-arg
"""Plot distribution as histogram or kernel density estimates."""
import numpy as np
import xarray as xr

from ..data import InferenceData
from ..rcparams import rcParams
from .plot_utils import get_plotting_function


def plot_dist(
    values,
    values2=None,
    color="C0",
    kind="auto",
    cumulative=False,
    label=None,
    rotated=False,
    rug=False,
    bw="default",
    quantiles=None,
    contour=True,
    fill_last=True,
    figsize=None,
    textsize=None,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    contour_kwargs=None,
    contourf_kwargs=None,
    pcolormesh_kwargs=None,
    hist_kwargs=None,
    is_circular=False,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
    **kwargs,
):
    """Plot distribution as histogram or kernel density estimates.

    By default continuous variables are plotted using KDEs and discrete ones using histograms

    Parameters
    ----------
    values : array-like
        Values to plot
    values2 : array-like, optional
        Values to plot. If present, a 2D KDE or a hexbin will be estimated
    color : string
        valid matplotlib color
    kind : string
        By default ("auto") continuous variables will use the kind defined by rcParam
        ``plot.density_kind`` and discrete ones will use histograms.
        To override this use "hist" to plot histograms and "kde" for KDEs
    cumulative : bool
        If true plot the estimated cumulative distribution function. Defaults to False.
        Ignored for 2D KDE
    label : string
        Text to include as part of the legend
    rotated : bool
        Whether to rotate the 1D KDE plot 90 degrees.
    rug : bool
        If True adds a rugplot. Defaults to False. Ignored for 2D KDE
    bw: Optional[float or str]
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "isj" or "experimental" when `is_circular` is False
        and "taylor" (for now) when `is_circular` is True.
        Defaults to "default" which means "experimental" when variable is not circular
        and "taylor" when it is.
    quantiles : list
        Quantiles in ascending order used to segment the KDE. Use [.25, .5, .75] for quartiles.
        Defaults to None.
    contour : bool
        If True plot the 2D KDE using contours, otherwise plot a smooth 2D KDE. Defaults to True.
    fill_last : bool
        If True fill the last contour of the 2D KDE plot. Defaults to True.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize. Not implemented for bokeh backend.
    plot_kwargs : dict
        Keywords passed to the pdf line of a 1D KDE.
    fill_kwargs : dict
        Keywords passed to the fill under the line (use fill_kwargs={'alpha': 0} to disable fill).
        Ignored for 2D KDE
    rug_kwargs : dict
        Keywords passed to the rug plot. Ignored if rug=False or for 2D KDE
        Use `space` keyword (float) to control the position of the rugplot. The larger this number
        the lower the rugplot.
    contour_kwargs : dict
        Keywords passed to the contourplot. Ignored for 1D KDE.
    contourf_kwargs : dict
        Keywords passed to ax.contourf. Ignored for 1D KDE.
    pcolormesh_kwargs : dict
        Keywords passed to ax.pcolormesh. Ignored for 1D KDE.
    hist_kwargs : dict
        Keywords passed to the histogram.
    is_circular : {False, True, "radians", "degrees"}. Default False.
        Select input type {"radians", "degrees"} for circular histogram or KDE plot. If True,
        default input type is "radians". When this argument is present, it interprets the
        values passed are from a circular variable measured in radians and a circular KDE is
        used. Inputs in "degrees" will undergo an internal conversion to radians. Only valid
        for 1D KDE. Defaults to False.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
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

    Examples
    --------
    Plot an integer distribution

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import arviz as az
        >>> a = np.random.poisson(4, 1000)
        >>> az.plot_dist(a)

    Plot a continuous distribution

    .. plot::
        :context: close-figs

        >>> b = np.random.normal(0, 1, 1000)
        >>> az.plot_dist(b)

    Add a rug under the Gaussian distribution

    .. plot::
        :context: close-figs

        >>> az.plot_dist(b, rug=True)

    Segment into quantiles

    .. plot::
        :context: close-figs

        >>> az.plot_dist(b, rug=True, quantiles=[.25, .5, .75])

    Plot as the cumulative distribution

    .. plot::
        :context: close-figs

        >>> az.plot_dist(b, rug=True, quantiles=[.25, .5, .75], cumulative=True)
    """
    values = np.asarray(values)

    if isinstance(values, (InferenceData, xr.Dataset)):
        raise ValueError(
            "InferenceData or xarray.Dataset object detected,"
            " use plot_posterior, plot_density or plot_pair"
            " instead of plot_dist"
        )

    if kind not in ["auto", "kde", "hist"]:
        raise TypeError(f'Invalid "kind":{kind}. Select from {{"auto","kde","hist"}}')

    if kind == "auto":
        kind = "hist" if values.dtype.kind == "i" else rcParams["plot.density_kind"]

    dist_plot_args = dict(
        # User Facing API that can be simplified
        values=values,
        values2=values2,
        color=color,
        kind=kind,
        cumulative=cumulative,
        label=label,
        rotated=rotated,
        rug=rug,
        bw=bw,
        quantiles=quantiles,
        contour=contour,
        fill_last=fill_last,
        figsize=figsize,
        textsize=textsize,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        rug_kwargs=rug_kwargs,
        contour_kwargs=contour_kwargs,
        contourf_kwargs=contourf_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs,
        hist_kwargs=hist_kwargs,
        ax=ax,
        backend_kwargs=backend_kwargs,
        is_circular=is_circular,
        show=show,
        **kwargs,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_dist", "distplot", backend)
    ax = plot(**dist_plot_args)
    return ax
