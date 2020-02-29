"""Plot posterior traces as violin plot."""
from ..data import convert_to_dataset
from .plot_utils import (
    _scale_fig_size,
    xarray_var_iter,
    filter_plotters_list,
    default_grid,
    get_plotting_function,
    matplotlib_kwarg_dealiaser,
)
from ..utils import _var_names
from ..rcparams import rcParams


def plot_violin(
    data,
    var_names=None,
    transform=None,
    quartiles=True,
    rug=False,
    credible_interval=None,
    shade=0.35,
    bw=4.5,
    sharex=True,
    sharey=True,
    figsize=None,
    textsize=None,
    ax=None,
    shade_kwargs=None,
    rug_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """Plot posterior of traces as violin plot.

    Notes
    -----
    If multiple chains are provided for a variable they will be combined

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names: list, optional
        List of variables to plot (defaults to None, which results in all variables plotted)
    transform : callable
        Function to transform data (defaults to None i.e. the identity function)
    quartiles : bool, optional
        Flag for plotting the interquartile range, in addition to the credible_interval*100%
        intervals. Defaults to True
    rug : bool
        If True adds a jittered rugplot. Defaults to False.
    credible_interval : float, optional
        Credible intervals. Defaults to 0.94.
    shade : float
        Alpha blending value for the shaded area under the curve, between 0
        (no shade) and 1 (opaque). Defaults to 0
    bw : float
        Bandwidth scaling factor. Should be larger than 0. The higher this number the smoother the
        KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule of thumb
        (the default rule used by SciPy).
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: int
        Text size of the point_estimates, axis ticks, and HPD. If None it will be autoscaled
        based on figsize.
    sharex : bool
        Defaults to True, violinplots share a common x-axis scale.
    sharey : bool
        Defaults to True, violinplots share a common y-axis scale.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    shade_kwargs : dicts, optional
        Additional keywords passed to `fill_between`, or `barh` to control the shade.
    rug_kwargs : dict
        Keywords passed to the rug plot. If true only the righ half side of the violin will be
        plotted.
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
    Show a default violin plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_violin(data)

    Show a default violin plot, but with a transformation applied to the data

    .. plot::
        :context: close-figs

        >>> az.plot_violin(data, var_names="tau", transform=np.log)

    """
    data = convert_to_dataset(data, group="posterior")
    if transform is not None:
        data = transform(data)
    var_names = _var_names(var_names, data)

    plotters = filter_plotters_list(
        list(xarray_var_iter(data, var_names=var_names, combined=True)), "plot_violin"
    )

    shade_kwargs = matplotlib_kwarg_dealiaser(shade_kwargs, "hexbin")

    rows, cols = default_grid(len(plotters))

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, _) = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    rug_kwargs = matplotlib_kwarg_dealiaser(rug_kwargs, "plot")

    if credible_interval is None:
        credible_interval = rcParams["stats.credible_interval"]
    else:
        if not 1 >= credible_interval > 0:
            raise ValueError("The value of credible_interval should be in the interval (0, 1]")

    violinplot_kwargs = dict(
        ax=ax,
        plotters=plotters,
        figsize=figsize,
        rows=rows,
        cols=cols,
        sharex=sharex,
        sharey=sharey,
        shade_kwargs=shade_kwargs,
        shade=shade,
        rug=rug,
        rug_kwargs=rug_kwargs,
        bw=bw,
        credible_interval=credible_interval,
        linewidth=linewidth,
        ax_labelsize=ax_labelsize,
        xt_labelsize=xt_labelsize,
        quartiles=quartiles,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    if backend == "bokeh":

        violinplot_kwargs.pop("ax_labelsize")
        violinplot_kwargs.pop("xt_labelsize")

        rug_kwargs.setdefault("fill_alpha", 0.1)
        rug_kwargs.setdefault("line_alpha", 0.1)

    else:
        rug_kwargs.setdefault("alpha", 0.1)
        rug_kwargs.setdefault("marker", ".")
        rug_kwargs.setdefault("linestyle", "")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_violin", "violinplot", backend)
    ax = plot(**violinplot_kwargs)
    return ax
