"""Plot posterior traces as violin plot."""

from ..data import convert_to_dataset
from ..labels import BaseLabeller
from ..sel_utils import xarray_var_iter
from ..utils import _var_names
from ..rcparams import rcParams
from .plot_utils import default_grid, filter_plotters_list, get_plotting_function


def plot_violin(
    data,
    var_names=None,
    combine_dims=None,
    filter_vars=None,
    transform=None,
    quartiles=True,
    rug=False,
    side="both",
    hdi_prob=None,
    shade=0.35,
    bw="default",
    circular=False,
    sharex=True,
    sharey=True,
    grid=None,
    figsize=None,
    textsize=None,
    labeller=None,
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
    data: obj
        Any object that can be converted to an :class:`arviz.InferenceData` object
        Refer to documentation of :func:`arviz.convert_to_dataset` for details
    var_names: list of variable names, optional
        Variables to be plotted, if None all variable are plotted. Prefix the
        variables by ``~`` when you want to exclude them from the plot.
    combine_dims : set_like of str, optional
        List of dimensions to reduce. Defaults to reducing only the "chain" and "draw" dimensions.
        See the :ref:`this section <common_combine_dims>` for usage examples.
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        ``pandas.filter``.
    transform: callable
        Function to transform data (defaults to None i.e. the identity function).
    quartiles: bool, optional
        Flag for plotting the interquartile range, in addition to the ``hdi_prob`` * 100%
        intervals. Defaults to ``True``.
    rug: bool
        If ``True`` adds a jittered rugplot. Defaults to ``False``.
    side : {"both", "left", "right"}, default "both"
        If ``both``, both sides of the violin plot are rendered. If ``left`` or ``right``, only
        the respective side is rendered. By separately plotting left and right halfs with
        different data, split violin plots can be achieved.
    hdi_prob: float, optional
        Plots highest posterior density interval for chosen percentage of density.
        Defaults to 0.94.
    shade: float
        Alpha blending value for the shaded area under the curve, between 0
        (no shade) and 1 (opaque). Defaults to 0.
    bw: float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "isj" or "experimental" when ``circular`` is ``False``
        and "taylor" (for now) when ``circular`` is ``True``.
        Defaults to "default" which means "experimental" when variable is not circular
        and "taylor" when it is.
    circular: bool, optional.
        If ``True``, it interprets `values` is a circular variable measured in radians
        and a circular KDE is used. Defaults to ``False``.
    grid : tuple
        Number of rows and columns. Defaults to None, the rows and columns are
        automatically inferred.
    figsize: tuple
        Figure size. If None it will be defined automatically.
    textsize: int
        Text size of the point_estimates, axis ticks, and highest density interval. If None it will
        be autoscaled based on ``figsize``.
    labeller : labeller instance, optional
        Class providing the method ``make_label_vert`` to generate the labels in the plot titles.
        Read the :ref:`label_guide` for more details and usage examples.
    sharex: bool
        Defaults to ``True``, violinplots share a common x-axis scale.
    sharey: bool
        Defaults to ``True``, violinplots share a common y-axis scale.
    ax: numpy array-like of matplotlib axes or bokeh figures, optional
        A 2D array of locations into which to plot the densities. If not supplied, Arviz will create
        its own array of plot areas (and return it).
    shade_kwargs: dicts, optional
        Additional keywords passed to :meth:`matplotlib.axes.Axes.fill_between`, or
        :meth:`matplotlib.axes.Axes.barh` to control the shade.
    rug_kwargs: dict
        Keywords passed to the rug plot. If true only the right half side of the violin will be
        plotted.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default to "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :func:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.
    show: bool, optional
        Call backend show function.

    Returns
    -------
    axes: matplotlib axes or bokeh figures

    See Also
    --------
    plot_forest: Forest plot to compare HDI intervals from a number of distributions.

    Examples
    --------
    Show a default violin plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_violin(data)

    """
    if labeller is None:
        labeller = BaseLabeller()

    data = convert_to_dataset(data, group="posterior")
    if transform is not None:
        data = transform(data)
    var_names = _var_names(var_names, data, filter_vars)

    plotters = filter_plotters_list(
        list(xarray_var_iter(data, var_names=var_names, combined=True, skip_dims=combine_dims)),
        "plot_violin",
    )

    rows, cols = default_grid(len(plotters), grid=grid)

    if hdi_prob is None:
        hdi_prob = rcParams["stats.ci_prob"]
    elif not 1 >= hdi_prob > 0:
        raise ValueError("The value of hdi_prob should be in the interval (0, 1]")

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
        side=side,
        bw=bw,
        textsize=textsize,
        labeller=labeller,
        circular=circular,
        hdi_prob=hdi_prob,
        quartiles=quartiles,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    if side not in ("both", "left", "right"):
        raise ValueError(f"'side' can only be 'both', 'left', or 'right', got: '{side}'")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_violin", "violinplot", backend)
    ax = plot(**violinplot_kwargs)
    return ax
