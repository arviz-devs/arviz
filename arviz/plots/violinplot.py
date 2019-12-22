"""Plot posterior traces as violin plot."""
from ..data import convert_to_dataset
from .plot_utils import (
    _scale_fig_size,
    xarray_var_iter,
    filter_plotters_list,
    default_grid,
    get_plotting_function,
)
from ..utils import _var_names


def plot_violin(
    data,
    var_names=None,
    quartiles=True,
    credible_interval=0.94,
    shade=0.35,
    bw=4.5,
    sharey=True,
    figsize=None,
    textsize=None,
    ax=None,
    kwargs_shade=None,
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
    quartiles : bool, optional
        Flag for plotting the interquartile range, in addition to the credible_interval*100%
        intervals. Defaults to True
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
    sharey : bool
        Defaults to True, violinplots share a common y-axis scale.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    kwargs_shade : dicts, optional
        Additional keywords passed to `fill_between`, or `barh` to control the shade.
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
    """
    data = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, data)

    plotters = filter_plotters_list(
        list(xarray_var_iter(data, var_names=var_names, combined=True)), "plot_violin"
    )

    if kwargs_shade is None:
        kwargs_shade = {}

    rows, cols = default_grid(len(plotters))

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, _) = _scale_fig_size(
        figsize, textsize, rows, cols
    )
    ax_labelsize *= 2

    violinplot_kwargs = dict(
        ax=ax,
        plotters=plotters,
        figsize=figsize,
        rows=rows,
        cols=cols,
        sharey=sharey,
        kwargs_shade=kwargs_shade,
        shade=shade,
        bw=bw,
        credible_interval=credible_interval,
        linewidth=linewidth,
        ax_labelsize=ax_labelsize,
        xt_labelsize=xt_labelsize,
        quartiles=quartiles,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend == "bokeh":

        violinplot_kwargs.pop("ax_labelsize")
        violinplot_kwargs.pop("xt_labelsize")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_violin", "violinplot", backend)
    ax = plot(**violinplot_kwargs)
    return ax
