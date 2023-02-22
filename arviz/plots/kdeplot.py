# pylint: disable=unexpected-keyword-arg
"""One-dimensional kernel density estimate plots."""
import warnings

import xarray as xr

from ..data import InferenceData
from ..rcparams import rcParams
from ..stats.density_utils import _fast_kde_2d, kde, _find_hdi_contours
from .plot_utils import get_plotting_function, _init_kwargs_dict


def plot_kde(
    values,
    values2=None,
    cumulative=False,
    rug=False,
    label=None,
    bw="default",
    adaptive=False,
    quantiles=None,
    rotated=False,
    contour=True,
    hdi_probs=None,
    fill_last=False,
    figsize=None,
    textsize=None,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    contour_kwargs=None,
    contourf_kwargs=None,
    pcolormesh_kwargs=None,
    is_circular=False,
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
    cumulative : bool, dafault False
        If True plot the estimated cumulative distribution function. Ignored for 2D KDE.
    rug : bool, default False
        Add a `rug plot <https://en.wikipedia.org/wiki/Rug_plot>`_ for a specific subset of
        values. Ignored for 2D KDE.
    label : string, optional
        Text to include as part of the legend.
    bw : float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "isj" or "experimental" when ``is_circular`` is False
        and "taylor" (for now) when ``is_circular`` is True.
        Defaults to "default" which means "experimental" when variable is not circular
        and "taylor" when it is.
    adaptive : bool, default False
        If True, an adaptative bandwidth is used. Only valid for 1D KDE.
    quantiles : list, optional
        Quantiles in ascending order used to segment the KDE. Use [.25, .5, .75] for quartiles.
    rotated : bool, default False
        Whether to rotate the 1D KDE plot 90 degrees.
    contour : bool, default True
        If True plot the 2D KDE using contours, otherwise plot a smooth 2D KDE.
    hdi_probs : list, optional
        Plots highest density credibility regions for the provided probabilities for a 2D KDE.
        Defaults to matplotlib chosen levels with no fixed probability associated.
    fill_last : bool, default False
        If True fill the last contour of the 2D KDE plot.
    figsize : (float, float), optional
        Figure size. If ``None`` it will be defined automatically.
    textsize : float, optional
        Text size scaling factor for labels, titles and lines. If ``None`` it will be autoscaled
        based on ``figsize``. Not implemented for bokeh backend.
    plot_kwargs : dict, optional
        Keywords passed to the pdf line of a 1D KDE. See :meth:`mpl:matplotlib.axes.Axes.plot`
        or :meth:`bokeh:bokeh.plotting.Figure.line` for a description of accepted values.
    fill_kwargs : dict, optional
        Keywords passed to the fill under the line (use ``fill_kwargs={'alpha': 0}``
        to disable fill). Ignored for 2D KDE. Passed to
        :meth:`bokeh.plotting.Figure.patch`.
    rug_kwargs : dict, optional
        Keywords passed to the rug plot. Ignored if ``rug=False`` or for 2D KDE
        Use ``space`` keyword (float) to control the position of the rugplot. The larger this number
        the lower the rugplot. Passed to :class:`bokeh:bokeh.models.glyphs.Scatter`.
    contour_kwargs : dict, optional
        Keywords passed to :meth:`mpl:matplotlib.axes.Axes.contour`
        to draw contour lines or :meth:`bokeh.plotting.Figure.patch`.
        Ignored for 1D KDE.
    contourf_kwargs : dict, optional
        Keywords passed to :meth:`mpl:matplotlib.axes.Axes.contourf`
        to draw filled contours. Ignored for 1D KDE.
    pcolormesh_kwargs : dict, optional
        Keywords passed to :meth:`mpl:matplotlib.axes.Axes.pcolormesh` or
        :meth:`bokeh.plotting.Figure.image`.
        Ignored for 1D KDE.
    is_circular : {False, True, "radians", "degrees"}. Default False
        Select input type {"radians", "degrees"} for circular histogram or KDE plot. If True,
        default input type is "radians". When this argument is present, it interprets ``values``
        as a circular variable measured in radians and a circular KDE is used. Inputs in
        "degrees" will undergo an internal conversion to radians.
    ax : axes, optional
        Matplotlib axes or bokeh figures.
    legend : bool, default True
        Add legend to the figure.
    backend : {"matplotlib", "bokeh"}, default "matplotlib"
        Select plotting backend.
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.
    show : bool, optional
        Call backend show function.
    return_glyph : bool, optional
        Internal argument to return glyphs for bokeh.

    Returns
    -------
    axes : matplotlib.Axes or bokeh.plotting.Figure
        Object containing the kde plot
    glyphs : list, optional
        Bokeh glyphs present in plot.  Only provided if ``return_glyph`` is True.

    See Also
    --------
    kde : One dimensional density estimation.
    plot_dist : Plot distribution as histogram or kernel density estimates.

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

    Plot KDE with adaptive bandwidth

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, adaptive=True)

    Plot KDE with a different bandwidth estimator

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, bw="scott")

    Plot KDE with a bandwidth specified manually

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, bw=0.4)

    Plot KDE for a circular variable

    .. plot::
        :context: close-figs

        >>> rvs = np.random.vonmises(mu=np.pi, kappa=2, size=500)
        >>> az.plot_kde(rvs, is_circular=True)


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


    Plot 2d contour KDE, without filling and contour lines using viridis cmap

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

    Plot 2d contour KDE with 30%, 60% and 90% HDI contours.

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, values2=tau_posterior, hdi_probs=[0.3, 0.6, 0.9])

    Plot 2d smooth KDE

    .. plot::
        :context: close-figs

        >>> az.plot_kde(mu_posterior, values2=tau_posterior, contour=False)

    """
    if isinstance(values, xr.Dataset):
        raise ValueError(
            "Xarray dataset object detected. Use plot_posterior, plot_density "
            "or plot_pair instead of plot_kde"
        )
    if isinstance(values, InferenceData):
        raise ValueError(
            " Inference Data object detected. Use plot_posterior "
            "or plot_pair instead of plot_kde"
        )

    if values2 is None:

        if bw == "default":
            bw = "taylor" if is_circular else "experimental"

        grid, density = kde(values, is_circular, bw=bw, adaptive=adaptive, cumulative=cumulative)
        lower, upper = grid[0], grid[-1]

        density_q = density if cumulative else density.cumsum() / density.sum()

        # This is just a hack placeholder for now
        xmin, xmax, ymin, ymax, gridsize = [None] * 5
    else:
        gridsize = (128, 128) if contour else (256, 256)
        density, xmin, xmax, ymin, ymax = _fast_kde_2d(values, values2, gridsize=gridsize)

        if hdi_probs is not None:
            # Check hdi probs are within bounds (0, 1)
            if min(hdi_probs) <= 0 or max(hdi_probs) >= 1:
                raise ValueError("Highest density interval probabilities must be between 0 and 1")

            # Calculate contour levels and sort for matplotlib
            contour_levels = _find_hdi_contours(density, hdi_probs)
            contour_levels.sort()

            contour_level_list = [0] + list(contour_levels) + [density.max()]

            # Add keyword arguments to contour, contourf
            contour_kwargs = _init_kwargs_dict(contour_kwargs)
            if "levels" in contour_kwargs:
                warnings.warn(
                    "Both 'levels' in contour_kwargs and 'hdi_probs' have been specified."
                    "Using 'hdi_probs' in favor of 'levels'.",
                    UserWarning,
                )
            contour_kwargs["levels"] = contour_level_list

            contourf_kwargs = _init_kwargs_dict(contourf_kwargs)
            if "levels" in contourf_kwargs:
                warnings.warn(
                    "Both 'levels' in contourf_kwargs and 'hdi_probs' have been specified."
                    "Using 'hdi_probs' in favor of 'levels'.",
                    UserWarning,
                )
            contourf_kwargs["levels"] = contour_level_list

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
        figsize=figsize,
        textsize=textsize,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        rug_kwargs=rug_kwargs,
        contour_kwargs=contour_kwargs,
        contourf_kwargs=contourf_kwargs,
        pcolormesh_kwargs=pcolormesh_kwargs,
        is_circular=is_circular,
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

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_kde", "kdeplot", backend)
    ax = plot(**kde_plot_args)

    return ax
