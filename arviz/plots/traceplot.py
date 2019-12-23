"""Plot kde or histograms and values from MCMC samples."""
from itertools import cycle
import warnings

import matplotlib.pyplot as plt

from .plot_utils import get_plotting_function, get_coords, xarray_var_iter
from ..data import convert_to_dataset
from ..utils import _var_names
from ..rcparams import rcParams


def plot_trace(
    data,
    var_names=None,
    coords=None,
    divergences="bottom",
    figsize=None,
    rug=False,
    lines=None,
    compact=False,
    combined=False,
    legend=False,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    hist_kwargs=None,
    trace_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """Plot distribution (histogram or kernel density estimates) and sampled values.

    If `divergences` data is available in `sample_stats`, will plot the location of divergences as
    dashed vertical lines.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : string, or list of strings
        One or more variables to be plotted.
    coords : mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    divergences : {"bottom", "top", None, False}
        Plot location of divergences on the traceplots. Options are "bottom", "top", or False-y.
    figsize : figure size tuple
        If None, size is (12, variables * 2)
    rug : bool
        If True adds a rugplot. Defaults to False. Ignored for 2D KDE.
        Only affects continuous variables.
    lines : tuple
        Tuple of (var_name, {'coord': selection}, [line, positions]) to be overplotted as
        vertical lines on the density and horizontal lines on the trace.
    compact : bool
        Plot multidimensional variables in a single plot.
    combined : bool
        Flag for combining multiple chains into a single line. If False (default), chains will be
        plotted separately.
    legend : bool
        Add a legend to the figure with the chain color code.
    plot_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    fill_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    rug_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
    hist_kwargs : dict
        Extra keyword arguments passed to `arviz.plot_dist`. Only affects discrete variables.
    trace_kwargs : dict
        Extra keyword arguments passed to `plt.plot`
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
    Plot a subset variables

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('non_centered_eight')
        >>> coords = {'school': ['Choate', 'Lawrenceville']}
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords)

    Show all dimensions of multidimensional variables in the same plot

    .. plot::
        :context: close-figs

        >>> az.plot_trace(data, compact=True)

    Combine all chains into one distribution

    .. plot::
        :context: close-figs

        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, combined=True)


    Plot reference lines against distribution and trace

    .. plot::
        :context: close-figs

        >>> lines = (('theta_t',{'school': "Choate"}, [-1]),)
        >>> az.plot_trace(data, var_names=('theta_t', 'theta'), coords=coords, lines=lines)

    """
    if divergences:
        try:
            divergence_data = convert_to_dataset(data, group="sample_stats").diverging
        except (ValueError, AttributeError):  # No sample_stats, or no `.diverging`
            divergences = False

    if coords is None:
        coords = {}

    if divergences:
        divergence_data = get_coords(
            divergence_data, {k: v for k, v in coords.items() if k in ("chain", "draw")}
        )
    else:
        divergence_data = False

    data = get_coords(convert_to_dataset(data, group="posterior"), coords)
    var_names = _var_names(var_names, data)

    if lines is None:
        lines = ()

    num_colors = len(data.chain) + 1 if combined else len(data.chain)

    # TODO: matplotlib is always required by arviz. Can we get rid of it?
    colors = [
        prop
        for _, prop in zip(
            range(num_colors), cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        )
    ]

    if compact:
        skip_dims = set(data.dims) - {"chain", "draw"}
    else:
        skip_dims = set()

    plotters = list(xarray_var_iter(data, var_names=var_names, combined=True, skip_dims=skip_dims))
    max_plots = rcParams["plot.max_subplots"]
    max_plots = len(plotters) if max_plots is None else max_plots
    if len(plotters) > max_plots:
        warnings.warn(
            "rcParams['plot.max_subplots'] ({max_plots}) is smaller than the number "
            "of variables to plot ({len_plotters}), generating only {max_plots} "
            "plots".format(max_plots=max_plots, len_plotters=len(plotters)),
            SyntaxWarning,
        )
        plotters = plotters[:max_plots]

    if figsize is None:
        figsize = (12, len(plotters) * 2)

    if trace_kwargs is None:
        trace_kwargs = {}
    trace_kwargs.setdefault("alpha", 0.35)

    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs.setdefault("alpha", 0.35)

    if plot_kwargs is None:
        plot_kwargs = {}
    if fill_kwargs is None:
        fill_kwargs = {}
    if rug_kwargs is None:
        rug_kwargs = {}

    # TODO: Check if this can be further simplified
    trace_plot_args = dict(
        # User Kwargs
        data=data,
        var_names=var_names,
        # coords = coords,
        divergences=divergences,
        figsize=figsize,
        rug=rug,
        lines=lines,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        rug_kwargs=rug_kwargs,
        hist_kwargs=hist_kwargs,
        trace_kwargs=trace_kwargs,
        # compact = compact,
        combined=combined,
        legend=legend,
        # Generated kwargs
        divergence_data=divergence_data,
        # skip_dims=skip_dims,
        plotters=plotters,
        colors=colors,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    plot = get_plotting_function("plot_trace", "traceplot", backend)
    axes = plot(**trace_plot_args)

    return axes
