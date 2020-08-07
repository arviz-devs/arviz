"""Plot a scatter, kde and/or hexbin of sampled parameters."""
import warnings
from typing import Optional, Union, List
import numpy as np

from ..data import convert_to_dataset, convert_to_inference_data
from .plot_utils import xarray_to_ndarray, get_plotting_function
from ..rcparams import rcParams
from ..utils import _var_names, get_coords


def plot_pair(
    data,
    group="posterior",
    var_names: Optional[List[str]] = None,
    filter_vars: Optional[str] = None,
    coords=None,
    marginals=False,
    figsize=None,
    textsize=None,
    kind: Union[str, List[str]] = "scatter",
    gridsize="auto",
    contour: Optional[bool] = None,
    plot_kwargs=None,
    fill_last=False,
    divergences=False,
    colorbar=False,
    ax=None,
    divergences_kwargs=None,
    scatter_kwargs=None,
    kde_kwargs=None,
    hexbin_kwargs=None,
    backend=None,
    backend_kwargs=None,
    marginal_kwargs=None,
    point_estimate=None,
    point_estimate_kwargs=None,
    point_estimate_marker_kwargs=None,
    reference_values=None,
    reference_values_kwargs=None,
    show=None,
):
    """
    Plot a scatter, kde and/or hexbin matrix with (optional) marginals on the diagonal.

    Parameters
    ----------
    data: obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    group: str, optional
        Specifies which InferenceData group should be plotted.  Defaults to 'posterior'.
    var_names: list of variable names, optional
        Variables to be plotted, if None all variable are plotted. Prefix the
        variables by `~` when you want to exclude them from the plot.
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    coords: mapping, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    marginals: bool, optional
        If True pairplot will include marginal distributions for every variable
    figsize: figure size tuple
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int
        Text size for labels. If None it will be autoscaled based on figsize.
    kind : str or List[str]
        Type of plot to display (scatter, kde and/or hexbin)
    gridsize: int or (int, int), optional
        Only works for kind=hexbin.
        The number of hexagons in the x-direction. The corresponding number of hexagons in the
        y-direction is chosen such that the hexagons are approximately regular.
        Alternatively, gridsize can be a tuple with two elements specifying the number of hexagons
        in the x-direction and the y-direction.
    contour : bool, optional, deprecated, Defaults to True.
        If True plot the 2D KDE using contours, otherwise plot a smooth 2D KDE. Defaults to True.
        **Note:** this default is implemented in the body of the code, not in argument processing.
    fill_last : bool
        If True fill the last contour of the 2D KDE plot. Defaults to True.
    divergences: Boolean
        If True divergences will be plotted in a different color, only if group is either 'prior'
        or 'posterior'.
    colorbar: bool
        If True a colorbar will be included as part of the plot (Defaults to False).
        Only works when kind=hexbin
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    divergences_kwargs: dicts, optional
        Additional keywords passed to ax.scatter for divergences
    scatter_kwargs:
        Additional keywords passed to ax.plot when using scatter kind
    kde_kwargs: dict, optional
        Additional keywords passed to az.plot_kde when using kde kind
    hexbin_kwargs: dict, optional
        Additional keywords passed to ax.hexbin when using hexbin kind
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    marginal_kwargs: dict, optional
        Additional keywords passed to az.plot_dist, modifying the marginal distributions
        plotted in the diagonal.
    point_estimate: str, optional
        Select point estimate from 'mean', 'mode' or 'median'. The point estimate will be
        plotted using a scatter marker and vertical/horizontal lines.
    point_estimate_kwargs: dict, optional
        Additional keywords passed to ax.vline, ax.hline (matplotlib) or ax.square, Span (bokeh)
    point_estimate_marker_kwargs: dict, optional
        Additional keywords passed to ax.scatter in point estimate plot. Not available in bokeh
    reference_values: dict, optional
        Reference values for the plotted variables. The Reference values will be plotted
        using a scatter marker
    reference_values_kwargs: dict, optional
        Additional keywords passed to ax.plot or ax.circle in reference values plot
    show: bool, optional
        Call backend show function.

    Returns
    -------
    axes: matplotlib axes or bokeh figures

    Examples
    --------
    KDE Pair Plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> centered = az.load_arviz_data('centered_eight')
        >>> coords = {'school': ['Choate', 'Deerfield']}
        >>> az.plot_pair(centered,
        >>>             var_names=['theta', 'mu', 'tau'],
        >>>             kind='kde',
        >>>             coords=coords,
        >>>             divergences=True,
        >>>             textsize=18)

    Hexbin pair plot

    .. plot::
        :context: close-figs

        >>> az.plot_pair(centered,
        >>>             var_names=['theta', 'mu'],
        >>>             coords=coords,
        >>>             textsize=18,
        >>>             kind='hexbin')

    Pair plot showing divergences and select variables with regular expressions

    .. plot::
        :context: close-figs

        >>> az.plot_pair(centered,
        ...             var_names=['^t', 'mu'],
        ...             filter_vars="regex",
        ...             coords=coords,
        ...             divergences=True,
        ...             textsize=18)
    """
    valid_kinds = ["scatter", "kde", "hexbin"]
    kind_boolean: Union[bool, List[bool]]
    if isinstance(kind, str):
        kind_boolean = kind in valid_kinds
    else:
        kind_boolean = [kind[i] in valid_kinds for i in range(len(kind))]
    if not np.all(kind_boolean):
        raise ValueError((f"Plot type {kind} not recognized." "Plot type must be in {valid_kinds}"))
    if fill_last or contour:
        warnings.warn(
            "fill_last and contour will be deprecated. Please use kde_kwargs", UserWarning,
        )
    if contour is None:
        contour = True

    if coords is None:
        coords = {}

    if plot_kwargs is None:
        plot_kwargs = {}
    else:
        warnings.warn(
            "plot_kwargs will be deprecated."
            " Please use scatter_kwargs, kde_kwargs and/or hexbin_kwargs",
            UserWarning,
        )

    if scatter_kwargs is None:
        scatter_kwargs = {}

    scatter_kwargs.setdefault("marker", ".")
    scatter_kwargs.setdefault("lw", 0)
    # Sets the default zorder higher than zorder of grid, which is 0.5
    scatter_kwargs.setdefault("zorder", 0.6)

    if kde_kwargs is None:
        kde_kwargs = {}

    if hexbin_kwargs is None:
        hexbin_kwargs = {}

    if divergences_kwargs is None:
        divergences_kwargs = {}

    divergences_kwargs.setdefault("marker", "o")
    divergences_kwargs.setdefault("markeredgecolor", "k")
    divergences_kwargs.setdefault("color", "C1")
    divergences_kwargs.setdefault("lw", 0)

    if marginal_kwargs is None:
        marginal_kwargs = {}

    if point_estimate_kwargs is None:
        point_estimate_kwargs = {}

    if point_estimate_marker_kwargs is None:
        point_estimate_marker_kwargs = {}

    # Get posterior draws and combine chains
    data = convert_to_inference_data(data)
    grouped_data = convert_to_dataset(data, group=group)
    var_names = _var_names(var_names, grouped_data, filter_vars)
    flat_var_names, infdata_group = xarray_to_ndarray(
        get_coords(grouped_data, coords), var_names=var_names, combined=True
    )

    divergent_data = None
    diverging_mask = None

    # Assigning divergence group based on group param
    if group == "posterior":
        divergent_group = "sample_stats"
    elif group == "prior":
        divergent_group = "sample_stats_prior"
    else:
        divergences = False

    # Get diverging draws and combine chains
    if divergences:
        if hasattr(data, divergent_group) and hasattr(getattr(data, divergent_group), "diverging"):
            divergent_data = convert_to_dataset(data, group=divergent_group)
            _, diverging_mask = xarray_to_ndarray(
                divergent_data, var_names=("diverging",), combined=True
            )
            diverging_mask = np.squeeze(diverging_mask)
        else:
            divergences = False
            warnings.warn(
                "Divergences data not found, plotting without divergences. "
                "Make sure the sample method provides divergences data and "
                "that it is present in the `diverging` field of `sample_stats` "
                "or `sample_stats_prior` or set divergences=False",
                UserWarning,
            )

    if gridsize == "auto":
        gridsize = int(len(infdata_group[0]) ** 0.35)

    numvars = len(flat_var_names)

    if numvars < 2:
        raise Exception("Number of variables to be plotted must be 2 or greater.")

    pairplot_kwargs = dict(
        ax=ax,
        infdata_group=infdata_group,
        numvars=numvars,
        figsize=figsize,
        textsize=textsize,
        kind=kind,
        plot_kwargs=plot_kwargs,
        scatter_kwargs=scatter_kwargs,
        kde_kwargs=kde_kwargs,
        hexbin_kwargs=hexbin_kwargs,
        contour=contour,
        fill_last=fill_last,
        gridsize=gridsize,
        colorbar=colorbar,
        divergences=divergences,
        diverging_mask=diverging_mask,
        divergences_kwargs=divergences_kwargs,
        flat_var_names=flat_var_names,
        backend_kwargs=backend_kwargs,
        marginal_kwargs=marginal_kwargs,
        show=show,
        marginals=marginals,
        point_estimate=point_estimate,
        point_estimate_kwargs=point_estimate_kwargs,
        point_estimate_marker_kwargs=point_estimate_marker_kwargs,
        reference_values=reference_values,
        reference_values_kwargs=reference_values_kwargs,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    if backend == "bokeh":
        pairplot_kwargs.pop("gridsize", None)
        pairplot_kwargs.pop("colorbar", None)
        pairplot_kwargs.pop("divergences_kwargs", None)
        pairplot_kwargs.pop("hexbin_values", None)
        pairplot_kwargs.pop("scatter_kwargs", None)
        point_estimate_kwargs.setdefault("line_color", "orange")
        point_estimate_marker_kwargs.setdefault("line_color", "orange")
    else:
        point_estimate_kwargs.setdefault("color", "C1")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_pair", "pairplot", backend)
    ax = plot(**pairplot_kwargs)
    return ax
