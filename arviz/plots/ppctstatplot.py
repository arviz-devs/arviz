"""T statistics Posterior/Prior predictive plot."""
from numbers import Integral
import platform
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


def plot_ppc_tstat(
    data,
    kind="kde",
    bpv = True,
    t_stat = "median",
    reference = None,
    n_ref = 100,
    hdi = 0.94,
    alpha=None,
    mean=True,
    color = 'C0',
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
    Plot T statistics for observed data and Posterior/Prior predictive

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
            raise TypeError(
                '`data` argument must have the group "{group}"'.format(group=groups)
            )

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

#y_pred = data.posterior_predictive[y_name].stack(dim=['chain', 'draw'])
#y_obs = data.observed_data[y_name]


######
#  backends
######

    from .kdeplot import plot_kde
    from .plot_utils import (make_label, _create_axes_grid)
    from ..numeric_utils import _fast_kde
    from scipy import stats

    if ax is None:
        fig, axes = _create_axes_grid(
            length_plotters, rows, cols, figsize=figsize, backend_kwargs=backend_kwargs
        )
    else:
        axes = np.ravel(ax)
        if len(axes) != length_plotters:
            raise ValueError(
                "Found {} variables to plot but {} axes instances. They must be equal.".format(
                    length_plotters, len(axes)
                )
            )

    for i, ax_i in enumerate(axes):
        var_name, selection, obs_vals = obs_plotters[i]
        pp_var_name, _, pp_vals = pp_plotters[i]
        dtype = predictive_dataset[pp_var_name].dtype.kind

        obs_vals = obs_vals.flatten()
        pp_vals = pp_vals.reshape(total_pp_samples, -1)

        if kind == "p_value":
            tstat_pit = np.mean(pp_vals <= obs_vals, axis=-1)
            tstat_pit_dens, xmin, xmax = _fast_kde(tstat_pit)
            x_s = np.linspace(xmin, xmax, len(tstat_pit_dens))
            ax_i.plot(x_s, tstat_pit_dens)

            if reference is not None:
                dist = stats.beta(obs_vals.size/2, obs_vals.size/2)
                if reference == "analytical":
                    lwb = dist.ppf((1-0.9999)/2)
                    upb = 1-lwb
                    x = np.linspace(lwb, upb, 500)
                    dens_ref = dist.pdf(x)
                    ax_i.plot(x, dens_ref, 'k--')
                elif reference == "samples":
                    x_ss, u_dens = sample_reference_distribution(dist, (n_ref, tstat_pit_dens.size, ))
                    ax_i.plot(x_ss, u_dens, alpha=0.1, color='C0') #**plot_ref_kwargs

        elif kind == "u_value":
            tstat_pit = np.mean(pp_vals <= obs_vals, axis=0)
            tstat_pit_dens, xmin, xmax = _fast_kde(tstat_pit)
            x_s = np.linspace(xmin, xmax, len(tstat_pit_dens))
            ax_i.plot(x_s, tstat_pit_dens)
            if reference is not None:
                if reference == "analytical":
                    n_obs = obs_vals.size
                    hdi_ = stats.beta(n_obs / 2, n_obs / 2).ppf((1 - hdi) / 2)
                    hdi_odds = (hdi_ / (1 - hdi_), (1 - hdi_) / hdi_)
                    ax_i.axhspan(*hdi_odds, alpha=0.35) #**plot_ref_kwargs
                elif reference == "samples":
                    dist = stats.uniform(0, 1)
                    x_ss, u_dens = sample_reference_distribution(dist, (tstat_pit_dens.size, n_ref))
                    ax_i.plot(x_ss, u_dens, alpha=0.1, color='C0') 
            ax_i.set_ylim(0, None)
            ax_i.set_xlim(0, 1)
        else:
            if isinstance(t_stat, str):
                if is_valid_quantile(t_stat):
                    t_stat = float(t_stat)
                    obs_vals = np.quantile(obs_vals, q=t_stat)
                    pp_vals = np.quantile(pp_vals, q=t_stat, axis=1)
                elif t_stat in ["mean", "median", "std"]:
                    if t_stat == 'mean':
                        tfunc = np.mean
                    elif t_stat == 'median':
                        tfunc = np.median
                    elif t_stat == 'std':
                        tfunc = np.std
                    obs_vals = tfunc(obs_vals)
                    pp_vals = tfunc(pp_vals, axis=1)
                else:
                    raise ValueError(f"T statistics {t_stat} not implemented")

            elif hasattr(t_stat, '__call__'):
                obs_vals = t_stat(obs_vals.flatten())
                pp_vals = t_stat(pp_vals)

            plot_kde(pp_vals, ax=ax_i)

            if bpv:
                p_value = np.mean(pp_vals <= obs_vals)
                ax_i.plot(0, 0, label=f'bpv={p_value:.2f}', alpha=0)
                ax_i.legend()

            if mean:
                ax_i.plot(obs_vals.mean(), 0, "o", color=color, markeredgecolor="k", markersize=markersize)



def is_valid_quantile(value):
    try:
        value = float(value)
        if 0 < value < 1:
            return True
        else:
            return False
    except ValueError:
        return False

from ..numeric_utils import _fast_kde
def sample_reference_distribution(dist, shape):
    x_ss = []
    densities = []
    dist_rvs = dist.rvs(size=shape)
    for idx in range(shape[1]):
        density, xmin, xmax = _fast_kde(dist_rvs[:,idx])
        x_s = np.linspace(xmin, xmax, len(density))
        x_ss.append(x_s)
        densities.append(density)
    return np.array(x_ss).T, np.array(densities).T
