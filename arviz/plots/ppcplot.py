"""Posterior predictive plot."""
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
    show=None,
):
    """
    Plot for posterior predictive checks.

    Parameters
    ----------
    data : az.InferenceData object
        InferenceData object containing the observed and posterior
        predictive data.
    kind : str
        Type of plot to display (kde, cumulative, or scatter). Defaults to kde.
    alpha : float
        Opacity of posterior predictive density curves. Defaults to 0.2 for kind = kde
        and cumulative, for scatter defaults to 0.7
    mean : bool
        Whether or not to plot the mean posterior predictive distribution. Defaults to True
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be
        autoscaled based on figsize.
    data_pairs : dict
        Dictionary containing relations between observed data and posterior predictive data.
        Dictionary structure:
        Key = data var_name
        Value = posterior predictive var_name
        For example, `data_pairs = {'y' : 'y_hat'}`
        If None, it will assume that the observed data and the posterior
        predictive data have the same variable name.
    var_names : list
        List of variables to be plotted. Defaults to all observed variables in the
        model if None.
    coords : dict
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Dimensions without a mapping specified will include all coordinates for
        that dimension. Defaults to including all coordinates for all
        dimensions if None.
    flatten : list
        List of dimensions to flatten in observed_data. Only flattens across the coordinates
        specified in the coords argument. Defaults to flattening all of the dimensions.
    flatten_pp : list
        List of dimensions to flatten in posterior_predictive. Only flattens across the coordinates
        specified in the coords argument. Defaults to flattening all of the dimensions.
        Dimensions should match flatten excluding dimensions for data_pairs parameters.
        If flatten is defined and flatten_pp is None, then `flatten_pp=flatten`.
    num_pp_samples : int
        The number of posterior predictive samples to plot. For `kind` = 'scatter' and
        `animation = False` if defaults to a maximum of 5 samples and will set jitter to 0.7
        unless defined otherwise. Otherwise it defaults to all provided samples.
    random_seed : int
        Random number generator seed passed to numpy.random.seed to allow
        reproducibility of the plot. By default, no seed will be provided
        and the plot will change each call if a random sample is specified
        by `num_pp_samples`.
    jitter : float
        If kind is "scatter", jitter will add random uniform noise to the height
        of the ppc samples and observed data. By default 0.
    animated : bool
        Create an animation of one posterior predictive sample per frame. Defaults to False.
    animation_kwargs : dict
        Keywords passed to `animation.FuncAnimation`.
    legend : bool
        Add legend to figure. By default True.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
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
    Plot the observed data KDE overlaid on posterior predictive KDEs.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('radon')
        >>> az.plot_ppc(data)

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
    for group in ("posterior_predictive", "observed_data"):
        if not hasattr(data, group):
            raise TypeError(
                '`data` argument must have the group "{group}" for ppcplot'.format(group=group)
            )

    if kind.lower() not in ("kde", "cumulative", "scatter"):
        raise TypeError("`kind` argument must be either `kde`, `cumulative`, or `scatter`")

    if data_pairs is None:
        data_pairs = {}

    if animation_kwargs is None:
        animation_kwargs = {}
    if platform.system() == "Linux":
        animation_kwargs.setdefault("blit", True)
    else:
        animation_kwargs.setdefault("blit", False)

    if animated and backend == "bokeh":
        raise TypeError("Animation option is only supported with matplotlib backend.")

    if animated and animation_kwargs["blit"] and platform.system() != "Linux":
        _log.warning(
            "If you experience problems rendering the animation try setting"
            "`animation_kwargs({'blit':False}) or changing the plotting backend (e.g. to TkAgg)"
        )

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
    posterior_predictive = data.posterior_predictive

    if var_names is None:
        var_names = list(observed.data_vars)
    var_names = _var_names(var_names, observed)
    pp_var_names = [data_pairs.get(var, var) for var in var_names]
    pp_var_names = _var_names(pp_var_names, posterior_predictive)

    if flatten_pp is None and flatten is None:
        flatten_pp = list(posterior_predictive.dims.keys())
    elif flatten_pp is None:
        flatten_pp = flatten
    if flatten is None:
        flatten = list(observed.dims.keys())

    if coords is None:
        coords = {}

    if random_seed is not None:
        np.random.seed(random_seed)

    total_pp_samples = posterior_predictive.sizes["chain"] * posterior_predictive.sizes["draw"]
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
                posterior_predictive.isel(coords),
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
        posterior_predictive=posterior_predictive,
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
        markersize=markersize,
        animation_kwargs=animation_kwargs,
        num_pp_samples=num_pp_samples,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend == "bokeh":

        ppcplot_kwargs.pop("animated")
        ppcplot_kwargs.pop("animation_kwargs")
        ppcplot_kwargs.pop("legend")
        ppcplot_kwargs.pop("xt_labelsize")
        ppcplot_kwargs.pop("ax_labelsize")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_ppc", "ppcplot", backend)
    axes = plot(**ppcplot_kwargs)
    return axes
