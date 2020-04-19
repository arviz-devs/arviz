"""Posterior-Prior plot."""

import numpy as np

from .plot_utils import (
    xarray_var_iter,
    get_coords,
    _scale_fig_size,
    get_plotting_function,
)
from ..utils import _var_names
from ..rcparams import rcParams


def plot_pp(
    data,
    figsize=None,
    textsize=None,
    var_names=None,
    coords=None,
    transform=None,
    data_pairs=None,
    legend=True,
    ax=None,
    prior_kwargs=None,
    posterior_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """
    Plot for posterior/Prior.

    Parameters
    ----------
    data : az.InferenceData object
        InferenceData object containing the posterior/prior data.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be
        autoscaled based on figsize.
    var_names : list
        List of variables to be plotted.
    coords : dict
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Dimensions without a mapping specified will include all coordinates for
        that dimension.
    transform : callable
        Function to transform data (defaults to None i.e. the identity function)
    data_pairs : dict
        Dictionary containing relations between prior data and posterior data.
        Dictionary structure:

        - key = prior data var_name
        - value = posterior var_name

        For example, `data_pairs = {'y' : 'y_hat'}`
        If None, it will assume that the prior data and the posterior
        data have the same variable name.
    legend : bool
        Add legend to figure. By default True.
    ax: axes, optional
        Matplotlib axes: a numpy 2d array of matplotlib axes.
    prior_kwargs : dicts, optional
        Additional keywords passed to `arviz.plot_kde` for prior group.
    posterior_kwargs : dicts, optional
        Additional keywords passed to `arviz.plot_kde` for posterior group.
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
    Plot the prior/posterior plot for specified vars and coords.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> data = az.load_arviz_data('radon')
        >>> az.plot_pp(data, var_names=["defs"], coords={"team" : ["Italy"]})

    """

    groups = ["prior", "posterior"]

    if transform is not None:
        data = transform(data)

    if coords is None:
        coords = {}

    if prior_kwargs is None:
        prior_kwargs = {}

    if posterior_kwargs is None:
        posterior_kwargs = {}

    if backend_kwargs is None:
        backend_kwargs = {}

    if data_pairs is None:
        data_pairs = {}

    datasets = [getattr(data, group) for group in groups]

    if var_names is None:
        var_names = list(datasets[0].data_vars)
    var_names = _var_names(var_names, datasets[0])
    vars = [var_names]
    for i in range(1, len(datasets)):
        var_name =[data_pairs.get(var, var) for var in var_names]
        var_name = _var_names(var_name,  datasets[i])
        vars.append(var_name)


    pp_plotters = [
        list(xarray_var_iter(get_coords(data, coords), var_names=var_names, combined=True))
        for data , var in zip(datasets, vars)
    ]

    nvars = len(pp_plotters[0])
    ngroups = len(groups)

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, 2 * nvars, ngroups
    )

    posterior_plot_kwargs = posterior_kwargs.pop("plot_kwargs", dict())
    posterior_plot_kwargs.setdefault("color", "red")
    posterior_plot_kwargs.setdefault("linewidth", linewidth)
    posterior_kwargs["plot_kwargs"] = posterior_plot_kwargs

    prior_plot_kwargs = prior_kwargs.pop("plot_kwargs", dict())
    prior_plot_kwargs.setdefault("color", "blue")
    prior_plot_kwargs.setdefault("linewidth", linewidth)
    prior_kwargs["plot_kwargs"] = prior_plot_kwargs

    ppplot_kwargs = dict(
        ax=ax,
        nvars=nvars,
        ngroups=ngroups,
        figsize=figsize,
        pp_plotters=pp_plotters,
        legend=legend,
        groups=groups,
        prior_kwargs=prior_kwargs,
        posterior_kwargs=posterior_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_pp", "ppplot", backend)
    axes = plot(**ppplot_kwargs)
    return axes
