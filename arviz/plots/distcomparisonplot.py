"""Density Comparison plot."""
import warnings
from ..labels import BaseLabeller
from ..rcparams import rcParams
from ..utils import _var_names, get_coords
from .plot_utils import get_plotting_function
from ..sel_utils import xarray_var_iter, xarray_sel_iter


def plot_dist_comparison(
    data,
    kind="latent",
    figsize=None,
    textsize=None,
    var_names=None,
    coords=None,
    combine_dims=None,
    transform=None,
    legend=True,
    labeller=None,
    ax=None,
    prior_kwargs=None,
    posterior_kwargs=None,
    observed_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    r"""Plot to compare fitted and unfitted distributions.

    The resulting plots will show the compared distributions both on
    separate axes (particularly useful when one of them is substantially tighter
    than another), and plotted together, displaying a grid of three plots per
    distribution.

    Parameters
    ----------
    data : InferenceData
        Any object that can be converted to an :class:`arviz.InferenceData` object
        containing the posterior/prior data. Refer to documentation of
        :func:`arviz.convert_to_dataset` for details.
    kind : {"latent", "observed"}, default "latent"
        kind of plot to display The "latent" option includes {"prior", "posterior"},
        and the "observed" option includes
        {"observed_data", "prior_predictive", "posterior_predictive"}.
    figsize : (float, float), optional
        Figure size. If ``None`` it will be defined automatically.
    textsize : float
        Text size scaling factor for labels, titles and lines. If ``None`` it will be
        autoscaled based on `figsize`.
    var_names : str, list, list of lists, optional
        if str, plot the variable. if list, plot all the variables in list
        of all groups. if list of lists, plot the vars of groups in respective lists.
        See :ref:`this section <common_var_names>` for usage examples.
    coords : dict
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Dimensions without a mapping specified will include all coordinates for
        that dimension. See :ref:`this section <common_coords>` for usage examples.
    combine_dims : set_like of str, optional
        List of dimensions to reduce. Defaults to reducing only the "chain" and "draw" dimensions.
        See :ref:`this section <common_combine_dims>` for usage examples.
    transform : callable
        Function to transform data (defaults to `None` i.e. the identity function).
    legend : bool
        Add legend to figure. By default True.
    labeller : Labeller, optional
        Class providing the method ``make_pp_label`` to generate the labels in the plot titles.
        Read the :ref:`label_guide` for more details and usage examples.
    ax : (nvars, 3) array-like of matplotlib_axes, optional
        Matplotlib axes: The ax argument should have shape (nvars, 3), where the
        last column is for the combined before/after plots and columns 0 and 1 are
        for the before and after plots, respectively.
    prior_kwargs : dicts, optional
        Additional keywords passed to :func:`arviz.plot_dist` for prior/predictive groups.
    posterior_kwargs : dicts, optional
        Additional keywords passed to :func:`arviz.plot_dist` for posterior/predictive groups.
    observed_kwargs : dicts, optional
        Additional keywords passed to :func:`arviz.plot_dist` for observed_data group.
    backend : {"matplotlib", "bokeh"}, default "matplotlib"
        Select plotting backend.
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
        For additional documentation check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : 2D ndarray of matplotlib_axes
        Returned object will have shape (nvars, 3),
        where the last column is the combined plot and the first columns are the single plots.

    See Also
    --------
    plot_bpv : Plot Bayesian p-value for observed data and Posterior/Prior predictive.

    Examples
    --------
    Plot the prior/posterior plot for specified vars and coords.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('rugby')
        >>> az.plot_dist_comparison(data, var_names=["defs"], coords={"team" : ["Italy"]})

    """
    all_groups = ["prior", "posterior"]

    if kind == "observed":
        all_groups = ["observed_data", "prior_predictive", "posterior_predictive"]

    if coords is None:
        coords = {}

    if labeller is None:
        labeller = BaseLabeller()

    datasets = []
    groups = []
    for group in all_groups:
        try:
            datasets.append(getattr(data, group))
            groups.append(group)
        except:  # pylint: disable=bare-except
            pass

    if var_names is None:
        var_names = list(datasets[0].data_vars)

    if isinstance(var_names, str):
        var_names = [var_names]

    if isinstance(var_names[0], str):
        var_names = [var_names for _ in datasets]

    var_names = [_var_names(vars, dataset) for vars, dataset in zip(var_names, datasets)]

    if transform is not None:
        datasets = [transform(dataset) for dataset in datasets]

    datasets = get_coords(datasets, coords)
    len_plots = rcParams["plot.max_subplots"] // (len(groups) + 1)
    len_plots = len_plots or 1
    dc_plotters = [
        list(xarray_var_iter(data, var_names=var, combined=True, skip_dims=combine_dims))[
            :len_plots
        ]
        for data, var in zip(datasets, var_names)
    ]

    total_plots = sum(
        1 for _ in xarray_sel_iter(datasets[0], var_names=var_names[0], combined=True)
    ) * (len(groups) + 1)
    maxplots = len(dc_plotters[0]) * (len(groups) + 1)

    if total_plots > rcParams["plot.max_subplots"]:
        warnings.warn(
            "rcParams['plot.max_subplots'] ({rcParam}) is smaller than the number "
            "of subplots to plot ({len_plotters}), generating only {max_plots} "
            "plots".format(
                rcParam=rcParams["plot.max_subplots"], len_plotters=total_plots, max_plots=maxplots
            ),
            UserWarning,
        )

    nvars = len(dc_plotters[0])
    ngroups = len(groups)

    distcomparisonplot_kwargs = dict(
        ax=ax,
        nvars=nvars,
        ngroups=ngroups,
        figsize=figsize,
        dc_plotters=dc_plotters,
        legend=legend,
        groups=groups,
        textsize=textsize,
        labeller=labeller,
        prior_kwargs=prior_kwargs,
        posterior_kwargs=posterior_kwargs,
        observed_kwargs=observed_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_dist_comparison", "distcomparisonplot", backend)
    axes = plot(**distcomparisonplot_kwargs)
    return axes
