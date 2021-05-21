"""Plot kde or histograms and values from MCMC samples."""
import warnings
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union, Sequence

from ..data import CoordSpec, InferenceData, convert_to_dataset
from ..labels import BaseLabeller
from ..rcparams import rcParams
from ..sel_utils import xarray_var_iter
from ..utils import _var_names, get_coords
from .plot_utils import KwargSpec, get_plotting_function


def plot_trace(
    data: InferenceData,
    var_names: Optional[Sequence[str]] = None,
    filter_vars: Optional[str] = None,
    transform: Optional[Callable] = None,
    coords: Optional[CoordSpec] = None,
    divergences: Optional[str] = "auto",
    kind: Optional[str] = "trace",
    figsize: Optional[Tuple[float, float]] = None,
    rug: bool = False,
    lines: Optional[List[Tuple[str, CoordSpec, Any]]] = None,
    circ_var_names: Optional[List[str]] = None,
    circ_var_units: str = "radians",
    compact: bool = True,
    compact_prop: Optional[Union[str, Mapping[str, Any]]] = None,
    combined: bool = False,
    chain_prop: Optional[Union[str, Mapping[str, Any]]] = None,
    legend: bool = False,
    plot_kwargs: Optional[KwargSpec] = None,
    fill_kwargs: Optional[KwargSpec] = None,
    rug_kwargs: Optional[KwargSpec] = None,
    hist_kwargs: Optional[KwargSpec] = None,
    trace_kwargs: Optional[KwargSpec] = None,
    rank_kwargs: Optional[KwargSpec] = None,
    labeller=None,
    axes=None,
    backend: Optional[str] = None,
    backend_config: Optional[KwargSpec] = None,
    backend_kwargs: Optional[KwargSpec] = None,
    show: Optional[bool] = None,
):
    """Plot distribution (histogram or kernel density estimates) and sampled values or rank plot.

    If `divergences` data is available in `sample_stats`, will plot the location of divergences as
    dashed vertical lines.

    Parameters
    ----------
    data: obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names: str or list of str, optional
        One or more variables to be plotted. Prefix the variables by `~` when you want
        to exclude them from the plot.
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    coords: dict of {str: slice or array_like}, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    divergences: {"bottom", "top", None}, optional
        Plot location of divergences on the traceplots.
    kind: {"trace", "rank_bar", "rank_vlines"}, optional
        Choose between plotting sampled values per iteration and rank plots.
    transform: callable, optional
        Function to transform data (defaults to None i.e.the identity function)
    figsize: tuple of (float, float), optional
        If None, size is (12, variables * 2)
    rug: bool, optional
        If True adds a rugplot of samples. Defaults to False. Ignored for 2D KDE.
        Only affects continuous variables.
    lines: list of tuple of (str, dict, array_like), optional
        List of (var_name, {'coord': selection}, [line, positions]) to be overplotted as
        vertical lines on the density and horizontal lines on the trace.
    circ_var_names : str or list of str, optional
        List of circular variables to account for when plotting KDE.
    circ_var_units : str
        Whether the variables in `circ_var_names` are in "degrees" or "radians".
    compact: bool, optional
        Plot multidimensional variables in a single plot.
    compact_prop: str or dict {str: array_like}, optional
        Tuple containing the property name and the property values to distinguish different
        dimensions with compact=True
    combined: bool, optional
        Flag for combining multiple chains into a single line. If False (default), chains will be
        plotted separately.
    chain_prop: str or dict {str: array_like}, optional
        Tuple containing the property name and the property values to distinguish different chains
    legend: bool, optional
        Add a legend to the figure with the chain color code.
    plot_kwargs, fill_kwargs, rug_kwargs, hist_kwargs: dict, optional
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    trace_kwargs: dict, optional
        Extra keyword arguments passed to `plt.plot`
    labeller : labeller instance, optional
        Class providing the method `make_label_vert` to generate the labels in the plot titles.
        Read the :ref:`label_guide` for more details and usage examples.
    rank_kwargs : dict, optional
        Extra keyword arguments passed to `arviz.plot_rank`
    axes: axes, optional
        Matplotlib axes or bokeh figures.
    backend: {"matplotlib", "bokeh"}, optional
        Select plotting backend.
    backend_config: dict, optional
        Currently specifies the bounds to use for bokeh axes. Defaults to value set in rcParams.
    backend_kwargs: dict, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show: bool, optional
        Call backend show function.

    Returns
    -------
    axes: matplotlib axes or bokeh figures

    Examples
    --------
    Plot a subset variables and select them with partial naming

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('non_centered_eight')
        >>> coords = {'school': ['Choate', 'Lawrenceville']}
        >>> az.plot_trace(data, var_names=('theta'), filter_vars="like", coords=coords)

    Show all dimensions of multidimensional variables in the same plot

    .. plot::
        :context: close-figs

        >>> az.plot_trace(data, compact=True)

    Display a rank plot instead of trace

    .. plot::
        :context: close-figs

        >>> az.plot_trace(data, var_names=["mu", "tau"], kind="rank_bars")

    Combine all chains into one distribution and select variables with regular expressions

    .. plot::
        :context: close-figs

        >>> az.plot_trace(
        >>>     data, var_names=('^theta'), filter_vars="regex", coords=coords, combined=True
        >>> )


    Plot reference lines against distribution and trace

    .. plot::
        :context: close-figs

        >>> lines = (('theta_t',{'school': "Choate"}, [-1]),)
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, lines=lines)

    """
    if kind not in {"trace", "rank_vlines", "rank_bars"}:
        raise ValueError("The value of kind must be either trace, rank_vlines or rank_bars.")

    if divergences == "auto":
        divergences = "top" if rug else "bottom"
    if divergences:
        try:
            divergence_data = convert_to_dataset(data, group="sample_stats").diverging
        except (ValueError, AttributeError):  # No sample_stats, or no `.diverging`
            divergences = None

    if coords is None:
        coords = {}

    if labeller is None:
        labeller = BaseLabeller()

    if divergences:
        divergence_data = get_coords(
            divergence_data, {k: v for k, v in coords.items() if k in ("chain", "draw")}
        )
    else:
        divergence_data = False

    coords_data = get_coords(convert_to_dataset(data, group="posterior"), coords)

    if transform is not None:
        coords_data = transform(coords_data)

    var_names = _var_names(var_names, coords_data, filter_vars)

    if compact:
        skip_dims = set(coords_data.dims) - {"chain", "draw"}
    else:
        skip_dims = set()

    plotters = list(
        xarray_var_iter(coords_data, var_names=var_names, combined=True, skip_dims=skip_dims)
    )
    max_plots = rcParams["plot.max_subplots"]
    max_plots = len(plotters) if max_plots is None else max(max_plots // 2, 1)
    if len(plotters) > max_plots:
        warnings.warn(
            "rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number "
            "of variables to plot ({len_plotters}), generating only {max_plots} "
            "plots".format(max_plots=max_plots, len_plotters=len(plotters)),
            UserWarning,
        )
        plotters = plotters[:max_plots]

    # TODO: Check if this can be further simplified
    trace_plot_args = dict(
        # User Kwargs
        data=coords_data,
        var_names=var_names,
        # coords = coords,
        divergences=divergences,
        kind=kind,
        figsize=figsize,
        rug=rug,
        lines=lines,
        circ_var_names=circ_var_names,
        circ_var_units=circ_var_units,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        rug_kwargs=rug_kwargs,
        hist_kwargs=hist_kwargs,
        trace_kwargs=trace_kwargs,
        rank_kwargs=rank_kwargs,
        compact=compact,
        compact_prop=compact_prop,
        combined=combined,
        chain_prop=chain_prop,
        legend=legend,
        labeller=labeller,
        # Generated kwargs
        divergence_data=divergence_data,
        # skip_dims=skip_dims,
        plotters=plotters,
        axes=axes,
        backend_config=backend_config,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_trace", "traceplot", backend)
    axes = plot(**trace_plot_args)

    return axes
