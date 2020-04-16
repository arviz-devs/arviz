"""Posterior-Prior plot."""

import numpy as np

from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    get_plotting_function,
)
from ..rcparams import rcParams


def plot_pp(
    data,
    figsize=None,
    textsize=None,
    var_names=None,
    coords=None,
    legend=True,
    ax=None,
    fill_kwargs=None,
    prior_plot_kwargs=None,
    posterior_plot_kwargs=None,
    prior_kwargs=None,
    posterior_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    """
    Plot for posterior-prior predictive.

    Parameters
    ----------
    data : az.InferenceData object
        InferenceData object containing the observed and posterior/prior predictive data.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    textsize: float
        Text size scaling factor for labels, titles and lines. If None it will be
        autoscaled based on figsize.
    var_names : list
        List of variables to be plotted. Defaults to all observed variables in the
        model if None.
    coords : dict
        Dictionary mapping dimensions to selected coordinates to be plotted.
        Dimensions without a mapping specified will include all coordinates for
        that dimension. Defaults to including all coordinates for all
        dimensions if None.
    legend : bool
        Add legend to figure. By default True.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    fill_kwargs : dicts, optional
        Additional keywords passed to `arviz.plot_kde` (to control the shade)
    plot_kwargs : dicts, optional
        Additional keywords passed to `arviz.plot_kde` or `plt.hist` (if type='hist')
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
        >>> az.plot_pp(data, var_names=["defs"], coords={"team" : ["Italy"]})

    """

    groups = ["prior", "posterior"]

    if coords is None:
        coords = {}

    if fill_kwargs is None:
        fill_kwargs = {}

    if prior_plot_kwargs is None:
        prior_plot_kwargs = {}

    if posterior_plot_kwargs is None:
        posterior_plot_kwargs = {}

    if prior_kwargs is None:
        prior_kwargs = {}

    if posterior_kwargs is None:
        posterior_kwargs = {}

    if backend_kwargs is None:
        backend_kwargs = {}

    pp_plotters = []
    for group in groups:

        group = getattr(data, group)
        coord = {}
        for key in coords.keys():
            coord[key] = np.where(np.in1d(group[key], coords[key]))[0]

        plotters = [
            tup for tup in xarray_var_iter(group.isel(coord), var_names=var_names, combined=True,)
        ]
        pp_plotters.append(plotters)

    ngroups = len(groups)
    nvars = len(pp_plotters[0])

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, 2 * nvars, ngroups
    )

    prior_plot_kwargs.setdefault("color", "blue")
    prior_plot_kwargs.setdefault("linewidth", linewidth)
    posterior_plot_kwargs.setdefault("color", "red")
    posterior_plot_kwargs.setdefault("linewidth", linewidth)

    ppplot_kwargs = dict(
        ax=ax,
        nvars=nvars,
        ngroups=ngroups,
        figsize=figsize,
        pp_plotters=pp_plotters,
        legend=legend,
        groups=groups,
        fill_kwargs=fill_kwargs,
        prior_plot_kwargs=prior_plot_kwargs,
        posterior_plot_kwargs=posterior_plot_kwargs,
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
