"""Plot quantile MC standard error."""
import numpy as np
import xarray as xr

from ..data import convert_to_dataset
from ..stats import mcse
from .plot_utils import (
    xarray_var_iter,
    default_grid,
    filter_plotters_list,
    get_plotting_function,
)
from ..rcparams import rcParams
from ..utils import _var_names, get_coords


def plot_mcse(
    idata,
    var_names=None,
    filter_vars=None,
    coords=None,
    errorbar=False,
    figsize=None,
    textsize=None,
    extra_methods=False,
    rug=False,
    rug_kind="diverging",
    n_points=20,
    ax=None,
    rug_kwargs=None,
    extra_kwargs=None,
    text_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
    **kwargs
):
    """Plot quantile or local Monte Carlo Standard Error.

    Parameters
    ----------
    idata: obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names: list of variable names, optional
        Variables to be plotted. Prefix the variables by `~` when you want to exclude
        them from the plot.
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    coords: dict, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    errorbar: bool, optional
        Plot quantile value +/- mcse instead of plotting mcse.
    figsize: tuple, optional
        Figure size. If None it will be defined automatically.
    textsize: float, optional
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    extra_methods: bool, optional
        Plot mean and sd MCSE as horizontal lines. Only taken into account when
        ``errorbar=False``.
    rug: bool
        Plot rug plot of values diverging or that reached the max tree depth.
    rug_kind: bool
        Variable in sample stats to use as rug mask. Must be a boolean variable.
    n_points: int
        Number of points for which to plot their quantile/local ess or number of subsets
        in the evolution plot.
    ax: numpy array-like of matplotlib axes or bokeh figures, optional
        A 2D array of locations into which to plot the densities. If not supplied, Arviz will create
        its own array of plot areas (and return it).
    rug_kwargs: dict
        kwargs passed to rug plot.
    extra_kwargs: dict, optional
        kwargs passed to ax.plot for extra methods lines.
    text_kwargs: dict, optional
        kwargs passed to ax.annotate for extra methods lines labels. It accepts the additional
        key ``x`` to set ``xy=(text_kwargs["x"], mcse)``
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

    References
    ----------
    * Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008

    Examples
    --------
    Plot quantile Monte Carlo Standard Error.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> idata = az.load_arviz_data("centered_eight")
        >>> coords = {"school": ["Deerfield", "Lawrenceville"]}
        >>> az.plot_mcse(
        ...     idata, var_names=["mu", "theta"], coords=coords
        ... )

    """
    mean_mcse = None
    sd_mcse = None

    if coords is None:
        coords = {}
    if "chain" in coords or "draw" in coords:
        raise ValueError("chain and draw are invalid coordinates for this kind of plot")

    data = get_coords(convert_to_dataset(idata, group="posterior"), coords)
    var_names = _var_names(var_names, data, filter_vars)

    probs = np.linspace(1 / n_points, 1 - 1 / n_points, n_points)
    mcse_dataset = xr.concat(
        [mcse(data, var_names=var_names, method="quantile", prob=p) for p in probs], dim="mcse_dim"
    )

    plotters = filter_plotters_list(
        list(xarray_var_iter(mcse_dataset, var_names=var_names, skip_dims={"mcse_dim"})),
        "plot_mcse",
    )
    length_plotters = len(plotters)
    rows, cols = default_grid(length_plotters)

    if extra_methods:
        mean_mcse = mcse(data, var_names=var_names, method="mean")
        sd_mcse = mcse(data, var_names=var_names, method="sd")

    mcse_kwargs = dict(
        ax=ax,
        plotters=plotters,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        figsize=figsize,
        errorbar=errorbar,
        rug=rug,
        data=data,
        probs=probs,
        kwargs=kwargs,
        extra_methods=extra_methods,
        mean_mcse=mean_mcse,
        sd_mcse=sd_mcse,
        textsize=textsize,
        text_kwargs=text_kwargs,
        rug_kwargs=rug_kwargs,
        extra_kwargs=extra_kwargs,
        idata=idata,
        rug_kind=rug_kind,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_mcse", "mcseplot", backend)
    ax = plot(**mcse_kwargs)
    return ax
