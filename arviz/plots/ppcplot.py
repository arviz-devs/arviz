"""Posterior/Prior predictive plot."""
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


def plot_ppc(
    data,
    kind="kde",
    alpha=None,
    mean=True,
    figsize=None,
    textsize=None,
    data_pairs=None,
    var_names=None,
    filter_vars=None,
    coords=None,
    flatten=None,
    flatten_pp=None,
    num_pp_samples=None,
    random_seed=None,
    jitter=None,
    animated=False,
    animation_kwargs=None,
    legend=True,
    ax=None,
    backend=None,
    backend_kwargs=None,
    group="posterior",
    show=None,
):
    """
    Plot for posterior/prior predictive checks.

    Parameters
    ----------
    data: az.InferenceData object
        InferenceData object containing the observed and posterior/prior predictive data.
    kind: str
        Type of plot to display (kde, cumulative, or scatter). Defaults to kde.
    alpha: float
        Opacity of posterior/prior predictive density curves.
        Defaults to 0.2 for kind = kde and cumulative, for scatter defaults to 0.7
    mean: bool
        Whether or not to plot the mean posterior/prior predictive distribution. Defaults to True
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
        Variables to be plotted, if `None` all variable are plotted. Prefix the
        variables by `~` when you want to exclude them from the plot.
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
    num_pp_samples: int
        The number of posterior/prior predictive samples to plot. For `kind` = 'scatter' and
        `animation = False` if defaults to a maximum of 5 samples and will set jitter to 0.7
        unless defined otherwise. Otherwise it defaults to all provided samples.
    random_seed: int
        Random number generator seed passed to numpy.random.seed to allow
        reproducibility of the plot. By default, no seed will be provided
        and the plot will change each call if a random sample is specified
        by `num_pp_samples`.
    jitter: float
        If kind is "scatter", jitter will add random uniform noise to the height
        of the ppc samples and observed data. By default 0.
    animated: bool
        Create an animation of one posterior/prior predictive sample per frame. Defaults to False.
        Only works with matploblib backend.
        To run animations inside a notebook you have to use the `nbAgg` matplotlib's backend.
        Try with `%matplotlib notebook` or  `%matplotlib  nbAgg`. You can switch back to the
        default matplotlib's backend with `%matplotlib  inline` or `%matplotlib  auto`.
        If switching back and forth between matplotlib's backend, you may need to run twice the cell
        with the animation.
        If you experience problems rendering the animation try setting
        `animation_kwargs({'blit':False}) or changing the matplotlib's backend (e.g. to TkAgg)
        If you run the animation from a script write `ax, ani = az.plot_ppc(.)`
    animation_kwargs : dict
        Keywords passed to `animation.FuncAnimation`. Ignored with matploblib backend.
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

    Examples
    --------
    Plot the observed data KDE overlaid on posterior predictive KDEs.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('radon')
        >>> az.plot_ppc(data, data_pairs={"obs":"obs"})
        >>> #az.plot_ppc(data, data_pairs={"obs":"obs_hat"})

    Plot the overlay with empirical CDFs.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, kind='cumulative')

    Use the coords and flatten parameters to plot selected variable dimensions
    across multiple plots.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, coords={'observed_county': ['ANOKA', 'BELTRAMI']}, flatten=[])

    Plot the overlay using a stacked scatter plot that is particularly useful
    when the sample sizes are small.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, kind='scatter', flatten=[],
        >>>             coords={'observed_county': ['AITKIN', 'BELTRAMI']})

    Plot random posterior predictive sub-samples.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, num_pp_samples=30, random_seed=7)
    """
    if group not in ("posterior", "prior"):
        raise TypeError("`group` argument must be either `posterior` or `prior`")

    for groups in ("{}_predictive".format(group), "observed_data"):
        if not hasattr(data, groups):
            raise TypeError(
                '`data` argument must have the group "{group}" for ppcplot'.format(group=groups)
            )

    if kind.lower() not in ("kde", "cumulative", "scatter"):
        raise TypeError("`kind` argument must be either `kde`, `cumulative`, or `scatter`")

    if data_pairs is None:
        data_pairs = {}

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    if animation_kwargs is None:
        animation_kwargs = {}
    if platform.system() == "Linux":
        animation_kwargs.setdefault("blit", True)
    else:
        animation_kwargs.setdefault("blit", False)

    if alpha is None:
        if animated:
            alpha = 1
        else:
            if kind.lower() == "scatter":
                alpha = 0.7
            else:
                alpha = 0.2

    if jitter is None:
        jitter = 0.0
    assert jitter >= 0.0

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

    if random_seed is not None:
        np.random.seed(random_seed)

    total_pp_samples = predictive_dataset.sizes["chain"] * predictive_dataset.sizes["draw"]
    if num_pp_samples is None:
        if kind == "scatter" and not animated:
            num_pp_samples = min(5, total_pp_samples)
        else:
            num_pp_samples = total_pp_samples

    if (
        not isinstance(num_pp_samples, Integral)
        or num_pp_samples < 1
        or num_pp_samples > total_pp_samples
    ):
        raise TypeError(
            "`num_pp_samples` must be an integer between 1 and "
            + "{limit}.".format(limit=total_pp_samples)
        )

    pp_sample_ix = np.random.choice(total_pp_samples, size=num_pp_samples, replace=False)

    for key in coords.keys():
        coords[key] = np.where(np.in1d(observed[key], coords[key]))[0]

    obs_plotters = filter_plotters_list(
        list(
            xarray_var_iter(
                observed.isel(coords), skip_dims=set(flatten), var_names=var_names, combined=True
            )
        ),
        "plot_ppc",
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

    ppcplot_kwargs = dict(
        ax=ax,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        figsize=figsize,
        animated=animated,
        obs_plotters=obs_plotters,
        pp_plotters=pp_plotters,
        predictive_dataset=predictive_dataset,
        pp_sample_ix=pp_sample_ix,
        kind=kind,
        alpha=alpha,
        linewidth=linewidth,
        mean=mean,
        xt_labelsize=xt_labelsize,
        ax_labelsize=ax_labelsize,
        jitter=jitter,
        total_pp_samples=total_pp_samples,
        legend=legend,
        group=group,
        markersize=markersize,
        animation_kwargs=animation_kwargs,
        num_pp_samples=num_pp_samples,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend == "bokeh":
        if animated:
            raise TypeError("Animation option is only supported with matplotlib backend.")

        ppcplot_kwargs.pop("animated")
        ppcplot_kwargs.pop("animation_kwargs")
        ppcplot_kwargs.pop("legend")
        ppcplot_kwargs.pop("group")
        ppcplot_kwargs.pop("xt_labelsize")
        ppcplot_kwargs.pop("ax_labelsize")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_ppc", "ppcplot", backend)
    axes = plot(**ppcplot_kwargs)
    return axes
