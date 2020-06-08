"""Bayesian p-value Posterior/Prior predictive plot."""
import logging
import numpy as np

from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    default_grid,
    filter_plotters_list,
    get_plotting_function,
)
from ..rcparams import rcParams
from ..utils import _var_names

_log = logging.getLogger(__name__)


def plot_bpv(
    data,
    kind="kde",
    bpv=True,
    t_stat="median",
    reference=None,
    n_ref=100,
    hdi=0.94,
    alpha=None,
    mean=True,
    color="C0",
    figsize=None,
    textsize=None,
    data_pairs=None,
    var_names=None,
    filter_vars=None,
    coords=None,
    flatten=None,
    flatten_pp=None,
    legend=True,
    ax=None,
    backend=None,
    backend_kwargs=None,
    group="posterior",
    show=None,
):
    """
    Plot Bayesian p-value for observed data and Posterior/Prior predictive

    Parameters
    ----------
    data: az.InferenceData object
        InferenceData object containing the observed and posterior/prior predictive data.
    kind: str
        Type of plot to display (kde, cumulative, or scatter). Defaults to kde.
    bpv : bool
        If True (default) add the bayesian p_value to the legend.
    alpha: float
        Opacity of posterior/prior predictive density curves.
        Defaults to 0.2 for kind = kde and cumulative, for scatter defaults to 0.7
    mean: bool
        Whether or not to plot the mean T statistic. Defaults to True
    color : str
        Matplotlib color
    figsize: tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be
        autoscaled based on figsize.
    data_pairs: dict
        Dictionary containing relations between observed data and posterior/prior predictive data.
        Dictionary structure:

        - key = data var_name
        - value = posterior/prior predictive var_name

        For example, `data_pairs = {'y' : 'y_hat'}`
        If None, it will assume that the observed data and the posterior/prior
        predictive data have the same variable name.
    var_names: list of variable names
        Variables to be plotted, if `None` all variable are plotted. Prefix the variables by `~`
        when you want to exclude them from the plot.
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    coords: dict
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Dimensions without a mapping specified will include all coordinates for
        that dimension. Defaults to including all coordinates for all
        dimensions if None.
    flatten: list
        List of dimensions to flatten in observed_data. Only flattens across the coordinates
        specified in the coords argument. Defaults to flattening all of the dimensions.
    flatten_pp: list
        List of dimensions to flatten in posterior_predictive/prior_predictive. Only flattens
        across the coordinates specified in the coords argument. Defaults to flattening all
        of the dimensions. Dimensions should match flatten excluding dimensions for data_pairs
        parameters. If flatten is defined and flatten_pp is None, then `flatten_pp=flatten`.
    legend : bool
        Add legend to figure. By default True.
    ax: numpy array-like of matplotlib axes or bokeh figures, optional
        A 2D array of locations into which to plot the densities. If not supplied, Arviz will create
        its own array of plot areas (and return it).
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    group: {"prior", "posterior"}, optional
        Specifies which InferenceData group should be plotted. Defaults to 'posterior'.
        Other value can be 'prior'.
    show: bool, optional
        Call backend show function.

    Returns
    -------
    axes: matplotlib axes or bokeh figures
    """
    if group not in ("posterior", "prior"):
        raise TypeError("`group` argument must be either `posterior` or `prior`")

    for groups in ("{}_predictive".format(group), "observed_data"):
        if not hasattr(data, groups):
            raise TypeError('`data` argument must have the group "{group}"'.format(group=groups))

    if kind.lower() not in ("t_stat", "u_value", "p_value"):
        raise TypeError("`kind` argument must be either `t_stat`, `u_value`, or `p_value`")

    if data_pairs is None:
        data_pairs = {}

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    observed = data.observed_data

    if group == "posterior":
        predictive_dataset = data.posterior_predictive
    elif group == "prior":
        predictive_dataset = data.prior_predictive

    if var_names is None:
        var_names = list(observed.data_vars)
    var_names = _var_names(var_names, observed, filter_vars)
    pp_var_names = [data_pairs.get(var, var) for var in var_names]
    pp_var_names = _var_names(pp_var_names, predictive_dataset, filter_vars)

    if flatten_pp is None and flatten is None:
        flatten_pp = list(predictive_dataset.dims.keys())
    elif flatten_pp is None:
        flatten_pp = flatten
    if flatten is None:
        flatten = list(observed.dims.keys())

    if coords is None:
        coords = {}

    total_pp_samples = predictive_dataset.sizes["chain"] * predictive_dataset.sizes["draw"]

    for key in coords.keys():
        coords[key] = np.where(np.in1d(observed[key], coords[key]))[0]

    obs_plotters = filter_plotters_list(
        list(
            xarray_var_iter(
                observed.isel(coords), skip_dims=set(flatten), var_names=var_names, combined=True
            )
        ),
        "plot_t_stats",
    )
    length_plotters = len(obs_plotters)
    pp_plotters = [
        tup
        for _, tup in zip(
            range(length_plotters),
            xarray_var_iter(
                predictive_dataset.isel(coords),
                var_names=pp_var_names,
                skip_dims=set(flatten_pp),
                combined=True,
            ),
        )
    ]
    rows, cols = default_grid(length_plotters)

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    bpvplot_kwargs = dict(
        ax=ax,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        obs_plotters=obs_plotters,
        pp_plotters=pp_plotters,
        predictive_dataset=predictive_dataset,
        total_pp_samples=total_pp_samples,
        kind=kind,
        bpv=bpv,
        t_stat=t_stat,
        reference=reference,
        n_ref=n_ref,
        hdi=hdi,
        alpha=alpha,
        mean=mean,
        color=color,
        figsize=figsize,
        xt_labelsize=xt_labelsize,
        ax_labelsize=ax_labelsize,
        markersize=markersize,
        linewidth=linewidth,
        flatten=flatten,
        legend=legend,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    # ppcplot_kwargs.pop("legend")
    # ppcplot_kwargs.pop("group")
    # ppcplot_kwargs.pop("xt_labelsize")
    # ppcplot_kwargs.pop("ax_labelsize")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_bpv", "bpvplot", backend)
    axes = plot(**bpvplot_kwargs)
    return axes
