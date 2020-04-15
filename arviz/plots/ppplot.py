"""Posterior-Prior plot."""
from numbers import Integral
import platform
import logging
import numpy as np

from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    get_plotting_function,
)
from ..rcparams import rcParams

_log = logging.getLogger(__name__)


def plot_pp(
    data,
    figsize=None,
    textsize=None,
    vars=None,
    coords=None,
    legend=True,
    ax=None,
    fill_kwargs=None,
    plot_kwargs=None,
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

        >>> import arvizz as az
        >>> data = az.load_arviz_data('radon')
        >>> az.plot_pp(data, vars=["defs"], coords={"team" : ["Italy"]})

    """

    groups = ["prior", "posterior"]

    if coords is None:
        coords = {}

    if fill_kwargs is None:
        fill_kwargs = {}

    if plot_kwargs is None:
        plot_kwargs = {}

    pp_plotters = []
    for group in groups:

        group = getattr(data, group)
        coord = {}
        for key in coords.keys():
            coord[key] = np.where(np.in1d(group[key], coords[key]))[0]

        plotters = [
            tup for tup in xarray_var_iter(group.isel(coord), var_names=vars, combined=True,)
        ]
        pp_plotters.append(plotters)

    cols = len(groups)
    rows = 2 * len(pp_plotters[0])

    (figsize, ax_labelsize, _, xt_labelsize, linewidth, markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )

    ppplot_kwargs = dict(
        ax=ax,
        length_plotters=(cols + 1) * rows,
        rows=rows,
        cols=cols,
        figsize=figsize,
        pp_plotters=pp_plotters,
        linewidth=linewidth,
        legend=legend,
        groups=groups,
        fill_kwargs=fill_kwargs,
        plot_kwargs=plot_kwargs,
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
