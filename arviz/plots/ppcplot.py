"""Posterior/Prior predictive plot."""

import logging
import warnings
from numbers import Integral

import numpy as np

from ..labels import BaseLabeller
from ..sel_utils import xarray_var_iter
from ..rcparams import rcParams
from ..utils import _var_names
from .plot_utils import default_grid, filter_plotters_list, get_plotting_function

_log = logging.getLogger(__name__)


def plot_ppc(
    data,
    kind="kde",
    alpha=None,
    mean=True,
    observed=None,
    observed_rug=False,
    color=None,
    colors=None,
    grid=None,
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
    labeller=None,
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
    data : InferenceData
        :class:`arviz.InferenceData` object containing the observed and posterior/prior
        predictive data.
    kind : str, default "kde"
        Type of plot to display ("kde", "cumulative", or "scatter").
    alpha : float, optional
        Opacity of posterior/prior predictive density curves.
        Defaults to 0.2 for ``kind = kde`` and cumulative, for scatter defaults to 0.7.
    mean : bool, default True
        Whether or not to plot the mean posterior/prior predictive distribution.
    observed : bool, optional
        Whether or not to plot the observed data. Defaults to True for ``group = posterior``
        and False for ``group = prior``.
    observed_rug : bool, default False
        Whether or not to plot a rug plot for the observed data. Only valid if `observed` is
        `True` and for kind `kde` or `cumulative`.
    color : list, optional
        List with valid matplotlib colors corresponding to the posterior/prior predictive
        distribution, observed data and mean of the posterior/prior predictive distribution.
        Defaults to ["C0", "k", "C1"].
    grid : tuple, optional
        Number of rows and columns. Defaults to None, the rows and columns are
        automatically inferred.
    figsize : tuple, optional
        Figure size. If None, it will be defined automatically.
    textsize : float, optional
        Text size scaling factor for labels, titles and lines. If None, it will be
        autoscaled based on ``figsize``.
    data_pairs : dict, optional
        Dictionary containing relations between observed data and posterior/prior predictive data.
        Dictionary structure:

        - key = data var_name
        - value = posterior/prior predictive var_name

        For example, ``data_pairs = {'y' : 'y_hat'}``
        If None, it will assume that the observed data and the posterior/prior
        predictive data have the same variable name.
    var_names : list of str, optional
        Variables to be plotted, if `None` all variable are plotted. Prefix the
        variables by ``~`` when you want to exclude them from the plot.
    filter_vars : {None, "like", "regex"}, default None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        ``pandas.filter``.
    coords : dict, optional
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Dimensions without a mapping specified will include all coordinates for
        that dimension. Defaults to including all coordinates for all
        dimensions if None.
    flatten : list
        List of dimensions to flatten in ``observed_data``. Only flattens across the coordinates
        specified in the ``coords`` argument. Defaults to flattening all of the dimensions.
    flatten_pp : list
        List of dimensions to flatten in posterior_predictive/prior_predictive. Only flattens
        across the coordinates specified in the ``coords`` argument. Defaults to flattening all
        of the dimensions. Dimensions should match flatten excluding dimensions for ``data_pairs``
        parameters. If ``flatten`` is defined and ``flatten_pp`` is None, then
        ``flatten_pp = flatten``.
    num_pp_samples : int
        The number of posterior/prior predictive samples to plot. For ``kind`` = 'scatter' and
        ``animation = False`` if defaults to a maximum of 5 samples and will set jitter to 0.7.
        unless defined. Otherwise it defaults to all provided samples.
    random_seed : int
        Random number generator seed passed to ``numpy.random.seed`` to allow
        reproducibility of the plot. By default, no seed will be provided
        and the plot will change each call if a random sample is specified
        by ``num_pp_samples``.
    jitter : float, default 0
        If ``kind`` is "scatter", jitter will add random uniform noise to the height
        of the ppc samples and observed data.
    animated : bool, default False
        Create an animation of one posterior/prior predictive sample per frame.
        Only works with matploblib backend.
        To run animations inside a notebook you have to use the `nbAgg` matplotlib's backend.
        Try with `%matplotlib notebook` or  `%matplotlib  nbAgg`. You can switch back to the
        default matplotlib's backend with `%matplotlib  inline` or `%matplotlib  auto`.
        If switching back and forth between matplotlib's backend, you may need to run twice the cell
        with the animation.
        If you experience problems rendering the animation try setting
        ``animation_kwargs({'blit':False})`` or changing the matplotlib's backend (e.g. to TkAgg)
        If you run the animation from a script write ``ax, ani = az.plot_ppc(.)``
    animation_kwargs : dict
        Keywords passed to  :class:`matplotlib.animation.FuncAnimation`. Ignored with
        matplotlib backend.
    legend : bool, default True
        Add legend to figure.
    labeller : labeller, optional
        Class providing the method ``make_pp_label`` to generate the labels in the plot titles.
        Read the :ref:`label_guide` for more details and usage examples.
    ax : numpy array-like of matplotlib_axes or bokeh figures, optional
        A 2D array of locations into which to plot the densities. If not supplied, Arviz will create
        its own array of plot areas (and return it).
    backend : str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default to "matplotlib".
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :func:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.
    group : {"prior", "posterior"}, optional
        Specifies which InferenceData group should be plotted. Defaults to 'posterior'.
        Other value can be 'prior'.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : matplotlib_axes or bokeh_figures
    ani : matplotlib.animation.FuncAnimation, optional
        Only provided if `animated` is ``True``.

    See Also
    --------
    plot_bpv : Plot Bayesian p-value for observed data and Posterior/Prior predictive.
    plot_loo_pit : Plot for posterior predictive checks using cross validation.
    plot_lm : Posterior predictive and mean plots for regression-like data.
    plot_ts : Plot timeseries data.

    Examples
    --------
    Plot the observed data KDE overlaid on posterior predictive KDEs.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('radon')
        >>> az.plot_ppc(data, data_pairs={"y":"y"})

    Plot the overlay with empirical CDFs.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, kind='cumulative')

    Use the ``coords`` and ``flatten`` parameters to plot selected variable dimensions
    across multiple plots. We will now modify the dimension ``obs_id`` to contain
    indicate the name of the county where the measure was taken. The change has to
    be done on both ``posterior_predictive`` and ``observed_data`` groups, which is
    why we will use :meth:`~arviz.InferenceData.map` to apply the same function to
    both groups. Afterwards, we will select the counties to be plotted with the
    ``coords`` arg.

    .. plot::
        :context: close-figs

        >>> obs_county = data.posterior["County"][data.constant_data["county_idx"]]
        >>> data = data.assign_coords(obs_id=obs_county, groups="observed_vars")
        >>> az.plot_ppc(data, coords={'obs_id': ['ANOKA', 'BELTRAMI']}, flatten=[])

    Plot the overlay using a stacked scatter plot that is particularly useful
    when the sample sizes are small.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, kind='scatter', flatten=[],
        >>>             coords={'obs_id': ['AITKIN', 'BELTRAMI']})

    Plot random posterior predictive sub-samples.

    .. plot::
        :context: close-figs

        >>> az.plot_ppc(data, num_pp_samples=30, random_seed=7)
    """
    if group not in ("posterior", "prior"):
        raise TypeError("`group` argument must be either `posterior` or `prior`")

    for groups in (f"{group}_predictive", "observed_data"):
        if not hasattr(data, groups):
            raise TypeError(f'`data` argument must have the group "{groups}" for ppcplot')

    if kind.lower() not in ("kde", "cumulative", "scatter"):
        raise TypeError("`kind` argument must be either `kde`, `cumulative`, or `scatter`")

    if colors is None:
        colors = ["C0", "k", "C1"]

    if isinstance(colors, str):
        raise TypeError("colors should be a list with 3 items.")

    if len(colors) != 3:
        raise ValueError("colors should be a list with 3 items.")

    if color is not None:
        warnings.warn("color has been deprecated in favor of colors", FutureWarning)
        colors[0] = color

    if data_pairs is None:
        data_pairs = {}

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()
    if backend == "bokeh" and animated:
        raise TypeError("Animation option is only supported with matplotlib backend.")

    observed_data = data.observed_data

    if group == "posterior":
        predictive_dataset = data.posterior_predictive
        if observed is None:
            observed = True
    elif group == "prior":
        predictive_dataset = data.prior_predictive
        if observed is None:
            observed = False

    if var_names is None:
        var_names = list(observed_data.data_vars)
    var_names = _var_names(var_names, observed_data, filter_vars)
    pp_var_names = [data_pairs.get(var, var) for var in var_names]
    pp_var_names = _var_names(pp_var_names, predictive_dataset, filter_vars)

    if flatten_pp is None:
        if flatten is None:
            flatten_pp = list(predictive_dataset.dims)
        else:
            flatten_pp = flatten
    if flatten is None:
        flatten = list(observed_data.dims)

    if coords is None:
        coords = {}
    else:
        coords = coords.copy()

    if labeller is None:
        labeller = BaseLabeller()

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
        raise TypeError(f"`num_pp_samples` must be an integer between 1 and {total_pp_samples}.")

    pp_sample_ix = np.random.choice(total_pp_samples, size=num_pp_samples, replace=False)

    for key in coords.keys():
        coords[key] = np.where(np.isin(observed_data[key], coords[key]))[0]

    obs_plotters = filter_plotters_list(
        list(
            xarray_var_iter(
                observed_data.isel(coords),
                skip_dims=set(flatten),
                var_names=var_names,
                combined=True,
                dim_order=["chain", "draw"],
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
                dim_order=["chain", "draw"],
            ),
        )
    ]
    rows, cols = default_grid(length_plotters, grid=grid)

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
        colors=colors,
        jitter=jitter,
        textsize=textsize,
        mean=mean,
        observed=observed,
        observed_rug=observed_rug,
        total_pp_samples=total_pp_samples,
        legend=legend,
        labeller=labeller,
        group=group,
        animation_kwargs=animation_kwargs,
        num_pp_samples=num_pp_samples,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_ppc", "ppcplot", backend)
    axes = plot(**ppcplot_kwargs)
    return axes
