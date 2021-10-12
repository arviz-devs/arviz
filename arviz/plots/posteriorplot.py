"""Plot posterior densities."""
from ..data import convert_to_dataset
from ..labels import BaseLabeller
from ..sel_utils import xarray_var_iter
from ..utils import _var_names, get_coords
from ..rcparams import rcParams
from .plot_utils import default_grid, filter_plotters_list, get_plotting_function


def plot_posterior(
    data,
    var_names=None,
    filter_vars=None,
    transform=None,
    coords=None,
    grid=None,
    figsize=None,
    textsize=None,
    hdi_prob=None,
    multimodal=False,
    skipna=False,
    round_to=None,
    point_estimate="auto",
    group="posterior",
    rope=None,
    ref_val=None,
    rope_color="C2",
    ref_val_color="C1",
    kind=None,
    bw="default",
    circular=False,
    bins=None,
    labeller=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
    **kwargs
):
    """Plot Posterior densities in the style of John K. Kruschke's book.

    Parameters
    ----------
    data: obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names: list of variable names
        Variables to be plotted, two variables are required. Prefix the variables by `~`
        when you want to exclude them from the plot.
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    transform: callable
        Function to transform data (defaults to None i.e.the identity function)
    coords: mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    grid : tuple
        Number of rows and columns. Defaults to None, the rows and columns are
        automatically inferred.
    figsize: tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    hdi_prob: float, optional
        Plots highest density interval for chosen percentage of density.
        Use 'hide' to hide the highest density interval. Defaults to 0.94.
    multimodal: bool
        If true (default) it may compute more than one credible interval if the distribution is
        multimodal and the modes are well separated.
    skipna : bool
        If true ignores nan values when computing the hdi and point estimates. Defaults to false.
    round_to: int, optional
        Controls formatting of floats. Defaults to 2 or the integer part, whichever is bigger.
    point_estimate: Optional[str]
        Plot point estimate per variable. Values should be 'mean', 'median', 'mode' or None.
        Defaults to 'auto' i.e. it falls back to default set in rcParams.
    group: str, optional
        Specifies which InferenceData group should be plotted. Defaults to ‘posterior’.
    rope: tuple or dictionary of tuples
        Lower and upper values of the Region Of Practical Equivalence. If a list is provided, its
        length should match the number of variables.
    ref_val: float or dictionary of floats
        display the percentage below and above the values in ref_val. Must be None (default),
        a constant, a list or a dictionary like see an example below. If a list is provided, its
        length should match the number of variables.
    rope_color: str, optional
        Specifies the color of ROPE and displayed percentage within ROPE
    ref_val_color: str, optional
        Specifies the color of the displayed percentage
    kind: str
        Type of plot to display (kde or hist) For discrete variables this argument is ignored and
        a histogram is always used. Defaults to rcParam ``plot.density_kind``
    bw: float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "isj" or "experimental" when `circular` is False
        and "taylor" (for now) when `circular` is True.
        Defaults to "default" which means "experimental" when variable is not circular
        and "taylor" when it is. Only works if `kind == kde`.
    circular: bool, optional
        If True, it interprets the values passed are from a circular variable measured in radians
        and a circular KDE is used. Only valid for 1D KDE. Defaults to False.
        Only works if `kind == kde`.
    bins: integer or sequence or 'auto', optional
        Controls the number of bins, accepts the same keywords `matplotlib.hist()` does. Only works
        if `kind == hist`. If None (default) it will use `auto` for continuous variables and
        `range(xmin, xmax + 1)` for discrete variables.
    labeller : labeller instance, optional
        Class providing the method `make_label_vert` to generate the labels in the plot titles.
        Read the :ref:`label_guide` for more details and usage examples.
    ax: numpy array-like of matplotlib axes or bokeh figures, optional
        A 2D array of locations into which to plot the densities. If not supplied, Arviz will create
        its own array of plot areas (and return it).
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show: bool, optional
        Call backend show function.
    **kwargs
        Passed as-is to plt.hist() or plt.plot() function depending on the value of `kind`.

    Returns
    -------
    axes: matplotlib axes or bokeh figures

    Examples
    --------
    Show a default kernel density plot following style of John Kruschke

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('centered_eight')
        >>> az.plot_posterior(data)

    Plot subset variables by specifying variable name exactly

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu'])

    Plot Region of Practical Equivalence (rope) and select variables with regular expressions

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu', '^the'], filter_vars="regex", rope=(-1, 1))

    Plot Region of Practical Equivalence for selected distributions

    .. plot::
        :context: close-figs

        >>> rope = {'mu': [{'rope': (-2, 2)}], 'theta': [{'school': 'Choate', 'rope': (2, 4)}]}
        >>> az.plot_posterior(data, var_names=['mu', 'theta'], rope=rope)

    Using `coords` argument to plot only a subset of data

    .. plot::
        :context: close-figs

        >>> coords = {"school": ["Choate","Phillips Exeter"]}
        >>> az.plot_posterior(data, var_names=["mu", "theta"], coords=coords)

    Add reference lines

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu', 'theta'], ref_val=0)

    Show point estimate of distribution

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu', 'theta'], point_estimate='mode')

    Show reference values using variable names and coordinates

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, ref_val= {"theta": [{"school": "Deerfield", "ref_val": 4},
        ...                                             {"school": "Choate", "ref_val": 3}]})

    Show reference values using a list

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, ref_val=[1] + [5] * 8 + [1])


    Plot posterior as a histogram

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu'], kind='hist')

    Change size of highest density interval

    .. plot::
        :context: close-figs

        >>> az.plot_posterior(data, var_names=['mu'], hdi_prob=.75)
    """
    data = convert_to_dataset(data, group=group)
    if transform is not None:
        data = transform(data)
    var_names = _var_names(var_names, data, filter_vars)

    if coords is None:
        coords = {}

    if labeller is None:
        labeller = BaseLabeller()

    if hdi_prob is None:
        hdi_prob = rcParams["stats.hdi_prob"]
    elif hdi_prob not in (None, "hide"):
        if not 1 >= hdi_prob > 0:
            raise ValueError("The value of hdi_prob should be in the interval (0, 1]")

    if point_estimate == "auto":
        point_estimate = rcParams["plot.point_estimate"]
    elif point_estimate not in {"mean", "median", "mode", None}:
        raise ValueError("The value of point_estimate must be either mean, median, mode or None.")

    if kind is None:
        kind = rcParams["plot.density_kind"]

    plotters = filter_plotters_list(
        list(xarray_var_iter(get_coords(data, coords), var_names=var_names, combined=True)),
        "plot_posterior",
    )
    length_plotters = len(plotters)
    rows, cols = default_grid(length_plotters, grid=grid)

    posteriorplot_kwargs = dict(
        ax=ax,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        figsize=figsize,
        plotters=plotters,
        bw=bw,
        circular=circular,
        bins=bins,
        kind=kind,
        point_estimate=point_estimate,
        round_to=round_to,
        hdi_prob=hdi_prob,
        multimodal=multimodal,
        skipna=skipna,
        textsize=textsize,
        ref_val=ref_val,
        rope=rope,
        ref_val_color=ref_val_color,
        rope_color=rope_color,
        labeller=labeller,
        kwargs=kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_posterior", "posteriorplot", backend)
    ax = plot(**posteriorplot_kwargs)
    return ax
