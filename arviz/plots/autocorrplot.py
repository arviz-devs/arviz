"""Autocorrelation plot of data."""

from ..data import convert_to_dataset
from ..labels import BaseLabeller
from ..sel_utils import xarray_var_iter
from ..rcparams import rcParams
from ..utils import _var_names, get_coords
from .plot_utils import default_grid, filter_plotters_list, get_plotting_function


def plot_autocorr(
    data,
    var_names=None,
    filter_vars=None,
    max_lag=None,
    combined=False,
    coords=None,
    grid=None,
    figsize=None,
    textsize=None,
    labeller=None,
    ax=None,
    backend=None,
    backend_config=None,
    backend_kwargs=None,
    show=None,
):
    r"""Bar plot of the autocorrelation function (ACF) for a sequence of data.

    The ACF plots are helpful as a convergence diagnostic for posteriors from MCMC
    samples which display autocorrelation.

    Parameters
    ----------
    data : InferenceData
        Any object that can be converted to an :class:`arviz.InferenceData` object
        refer to documentation of :func:`arviz.convert_to_dataset` for details
    var_names : list of str, optional
        Variables to be plotted. Prefix the variables by ``~`` when you want to exclude
        them from the plot. See :ref:`this section <common_var_names>` for usage examples.
    filter_vars : {None, "like", "regex"}, default None
        If `None` (default), interpret `var_names` as the real variables names. If "like",
        interpret `var_names` as substrings of the real variables names. If "regex",
        interpret `var_names` as regular expressions on the real variables names. See
        :ref:`this section <common_filter_vars>` for usage examples.
    coords: mapping, optional
        Coordinates of var_names to be plotted. Passed to :meth:`xarray.Dataset.sel`
    max_lag : int, optional
        Maximum lag to calculate autocorrelation. By Default, the plot displays the
        first 100 lag or the total number of draws, whichever is smaller.
    combined : bool, default False
        Flag for combining multiple chains into a single chain. If False, chains will be
        plotted separately.
    grid : tuple, optional
        Number of rows and columns. Defaults to None, the rows and columns are
        automatically inferred. See :ref:`this section <common_grid>` for usage examples.
    figsize : (float, float), optional
        Figure size. If None it will be defined automatically.
        Note this is not used if `ax` is supplied.
    textsize : float, optional
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on `figsize`.
    labeller : Labeller, optional
        Class providing the method ``make_label_vert`` to generate the labels in the plot titles.
        Read the :ref:`label_guide` for more details and usage examples.
    ax : 2D array-like of matplotlib_axes or bokeh_figure, optional
        A 2D array of locations into which to plot the densities. If not supplied, ArviZ will create
        its own array of plot areas (and return it).
    backend : {"matplotlib", "bokeh"}, default "matplotlib"
        Select plotting backend.
    backend_config : dict, optional
        Currently specifies the bounds to use for bokeh axes. Defaults to value set in ``rcParams``.
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : matplotlib_axes or bokeh_figures

    See Also
    --------
    autocov : Compute autocovariance estimates for every lag for the input array.
    autocorr : Compute autocorrelation using FFT for every lag for the input array.

    Examples
    --------
    Plot default autocorrelation

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_autocorr(data)

    Plot subset variables by specifying variable name exactly

    .. plot::
        :context: close-figs

        >>> az.plot_autocorr(data, var_names=['mu', 'tau'] )


    Combine chains by variable and select variables by excluding some with partial naming

    .. plot::
        :context: close-figs

        >>> az.plot_autocorr(data, var_names=['~thet'], filter_vars="like", combined=True)


    Specify maximum lag (x axis bound)

    .. plot::
        :context: close-figs

        >>> az.plot_autocorr(data, var_names=['mu', 'tau'], max_lag=200, combined=True)
    """
    data = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, data, filter_vars)

    # Default max lag to 100 or max length of chain
    if max_lag is None:
        max_lag = min(100, data["draw"].shape[0])

    if coords is None:
        coords = {}

    if labeller is None:
        labeller = BaseLabeller()

    plotters = filter_plotters_list(
        list(
            xarray_var_iter(
                get_coords(data, coords), var_names, combined, dim_order=["chain", "draw"]
            )
        ),
        "plot_autocorr",
    )
    rows, cols = default_grid(len(plotters), grid=grid)

    autocorr_plot_args = dict(
        axes=ax,
        plotters=plotters,
        max_lag=max_lag,
        figsize=figsize,
        rows=rows,
        cols=cols,
        combined=combined,
        textsize=textsize,
        labeller=labeller,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    if backend == "bokeh":
        autocorr_plot_args.update(backend_config=backend_config)

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_autocorr", "autocorrplot", backend)
    axes = plot(**autocorr_plot_args)

    return axes
