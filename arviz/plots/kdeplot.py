# pylint: disable=unexpected-keyword-arg
"""One-dimensional kernel density estimate plots."""
import xarray as xr

from ..data import InferenceData
from ..numeric_utils import _fast_kde, _fast_kde_2d
from .plot_utils import get_plotting_function
from ..rcparams import rcParams


def plot_kde(
    values,
    values2=None,
    cumulative=False,
    rug=False,
    label=None,
    bw=4.5,
    quantiles=None,
    rotated=False,
    contour=True,
    fill_last=False,
    textsize=None,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    contour_kwargs=None,
    contourf_kwargs=None,
    pcolormesh_kwargs=None,
    ax=None,
    legend=True,
    backend=None,
    backend_kwargs=None,
    show=None,
    return_glyph=False,
    **kwargs
):
    """1D or 2D KDE plot taking into account boundary conditions.

    Parameters
    ----------
    values : array-like
        Values to plot
    values2 : array-like, optional
        Values to plot. If present, a 2D KDE will be estimated
    cumulative : bool
        If true plot the estimated cumulative distribution function. Defaults to False.
        Ignored for 2D KDE
    rug : bool
        If True adds a rugplot. Defaults to False. Ignored for 2D KDE
    label : string
        Text to include as part of the legend
    bw : float
        Bandwidth scaling factor for 1D KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's
        rule of thumb (the default rule used by SciPy).
    quantiles : list
        Quantiles in ascending order used to segment the KDE. Use [.25, .5, .75] for quartiles.
        Defaults to None.
    rotated : bool
        Whether to rotate the 1D KDE plot 90 degrees.
    contour : bool
        If True plot the 2D KDE using contours, otherwise plot a smooth 2D KDE. Defaults to True.
    fill_last : bool
        If True fill the last contour of the 2D KDE plot. Defaults to False.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize. Not implemented for bokeh backend.
    plot_kwargs : dict
        Keywords passed to the pdf line of a 1D KDE. See :meth:`mpl:matplotlib.axes.Axes.plot`
        or :meth:`bokeh:bokeh.plotting.figure.Figure.line` for a description of accepted values.
    fill_kwargs : dict
        Keywords passed to the fill under the line (use fill_kwargs={'alpha': 0} to disable fill).
        Ignored for 2D KDE
    rug_kwargs : dict
        Keywords passed to the rug plot. Ignored if rug=False or for 2D KDE
        Use `space` keyword (float) to control the position of the rugplot. The larger this number
        the lower the rugplot.
    contour_kwargs : dict
        Keywords passed to ax.contour to draw contour lines. Ignored for 1D KDE.
    contourf_kwargs : dict
        Keywords passed to ax.contourf to draw filled contours. Ignored for 1D KDE.
    pcolormesh_kwargs : dict
        Keywords passed to ax.pcolormesh. Ignored for 1D KDE.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    legend : bool
        Add legend to the figure. By default True.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show : bool, optional
        Call backend show function.
    return_glyph : bool, optional
        Internal argument to return glyphs for bokeh

    Returns
    -------
    axes : matplotlib.Axes or bokeh.plotting.Figure
        Object containing the kde plot
    glyphs : list, optional
        Bokeh glyphs present in plot.  Only provided if ``return_glyph`` is True.

    Examples
    --------
    Plot default KDE

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> non_centered = az.load_arviz_data('non_centered_eight')
        >>> mu_posterior = np.concatenate(non_centered.posterior["mu"].values)
        >>> tau_posterior = np.concatenate(non_centered.posterior["tau"].values)
        >>> az.plot_kde(mu_posterior)


    Plot KDE with rugplot

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, rug=True)


    Plot a cumulative distribution

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, cumulative=True)



    Rotate plot 90 degrees

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, rotated=True)


    Plot 2d contour KDE

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, values2=tau_posterior)


    Plot 2d contour KDE, without filling and countour lines using viridis cmap

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, values2=tau_posterior,
        ...             contour_kwargs={"colors":None, "cmap":plt.cm.viridis},
        ...             contourf_kwargs={"alpha":0});

    Plot 2d contour KDE, set the number of levels to 3.

    .. plot::
        :context: close-figs

        >>> az.plot_kde(
        ...     mu_posterior, values2=tau_posterior,
        ...     contour_kwargs={"levels":3}, contourf_kwargs={"levels":3}
        ... );

    Plot 2d smooth KDE

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, values2=tau_posterior, contour=False)

    """
    if isinstance(values, xr.Dataset):
        raise ValueError(
            "Xarray dataset object detected.Use plot_posterior, plot_density, plot_joint"
            "or plot_pair instead of plot_kde"
        )
    if isinstance(values, InferenceData):
        raise ValueError(
            " Inference Data object detected. Use plot_posterior "
            "or plot_pair instead of plot_kde"
        )

    if values2 is None:
        density, lower, upper = _fast_kde(values, cumulative, bw)

        if cumulative:
            density_q = density
        else:
            density_q = density.cumsum() / density.sum()

        # This is just a hack placeholder for now
        xmin, xmax, ymin, ymax, gridsize = [None] * 5
    else:
        gridsize = (128, 128) if contour else (256, 256)
        density, xmin, xmax, ymin, ymax = _fast_kde_2d(values, values2, gridsize=gridsize)

        lower, upper, density_q = [None] * 3

    kde_plot_args = dict(
        # Internal API
        density=density,
        lower=lower,
        upper=upper,
        density_q=density_q,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        gridsize=gridsize,
        # User Facing API that can be simplified
        values=values,
        values2=values2,
        rug=rug,
        label=label,
        quantiles=quantiles,
        rotated=rotated,
        contour=contour,
        fill_last=fill_last,
        textsize=textsize,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        rug_kwargs=rug_kwargs,
        contour_kwargs=contour_kwargs,
        contourf_kwargs=contourf_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs,
        ax=ax,
        legend=legend,
        backend_kwargs=backend_kwargs,
        show=show,
        return_glyph=return_glyph,
        **kwargs,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    if backend == "bokeh":
        kde_plot_args.pop("textsize")
        kde_plot_args.pop("label")
        kde_plot_args.pop("legend")
    else:
        kde_plot_args.pop("return_glyph")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_kde", "kdeplot", backend)
    ax = plot(**kde_plot_args)

    return ax
