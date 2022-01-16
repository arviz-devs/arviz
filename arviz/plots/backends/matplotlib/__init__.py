# pylint: disable=wrong-import-position
"""Matplotlib Plotting Backend."""
import matplotlib as mpl

from matplotlib.cbook import normalize_kwargs
from matplotlib.pyplot import subplots
from numpy import ndenumerate

from ....rcparams import rcParams


def backend_kwarg_defaults(*args, **kwargs):
    """Get default kwargs for backend.

    For args add a tuple with key and rcParam key pair.
    """
    defaults = {**kwargs}
    # add needed default args from arviz.rcParams
    for key, arg in args:
        defaults.setdefault(key, rcParams[arg])
    return defaults


def backend_show(show):
    """Set default behaviour for show if not explicitly defined."""
    if show is None:
        show = rcParams["plot.matplotlib.show"]
    return show


def create_axes_grid(length_plotters, rows=1, cols=1, backend_kwargs=None):
    """Create figure and axes for grids with multiple plots.

    Parameters
    ----------
    length_plotters : int
        Number of axes required
    rows : int
        Number of rows
    cols : int
        Number of columns
    backend_kwargs: dict, optional
        kwargs for backend figure.

    Returns
    -------
    fig : matplotlib figure
    ax : matplotlib axes
    """
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {**backend_kwarg_defaults(), **backend_kwargs}

    fig, axes = subplots(rows, cols, **backend_kwargs)
    extra = (rows * cols) - length_plotters
    if extra > 0:
        for (row, col), ax in ndenumerate(axes):
            if (row * cols + col + 1) > length_plotters:
                ax.set_axis_off()
    return fig, axes


def matplotlib_kwarg_dealiaser(args, kind):
    """De-aliase the kwargs passed to plots."""
    if args is None:
        return {}
    matplotlib_kwarg_dealiaser_dict = {
        "scatter": mpl.collections.PathCollection,
        "plot": mpl.lines.Line2D,
        "hist": mpl.patches.Patch,
        "bar": mpl.patches.Rectangle,
        "hexbin": mpl.collections.PolyCollection,
        "fill_between": mpl.collections.PolyCollection,
        "hlines": mpl.collections.LineCollection,
        "text": mpl.text.Text,
        "contour": mpl.contour.ContourSet,
        "pcolormesh": mpl.collections.QuadMesh,
    }
    return normalize_kwargs(args, getattr(matplotlib_kwarg_dealiaser_dict[kind], "_alias_map", {}))


def dealiase_sel_kwargs(kwargs, prop_dict, idx):
    """Generate kwargs dict from kwargs and prop_dict.

    Gets property at position ``idx`` for each property in prop_dict and adds it to
    ``kwargs``. Values in prop_dict are dealiased and overwrite values in
    kwargs with the same key .

    Parameters
    ----------
    kwargs : dict
    prop_dict : dict of {str : array_like}
    idx : int
    """
    return {
        **kwargs,
        **matplotlib_kwarg_dealiaser(
            {prop: props[idx] for prop, props in prop_dict.items()}, "plot"
        ),
    }


from .autocorrplot import plot_autocorr
from .bpvplot import plot_bpv
from .compareplot import plot_compare
from .densityplot import plot_density
from .distplot import plot_dist
from .elpdplot import plot_elpd
from .energyplot import plot_energy
from .essplot import plot_ess
from .forestplot import plot_forest
from .hdiplot import plot_hdi
from .kdeplot import plot_kde
from .khatplot import plot_khat
from .loopitplot import plot_loo_pit
from .mcseplot import plot_mcse
from .pairplot import plot_pair
from .parallelplot import plot_parallel
from .posteriorplot import plot_posterior
from .ppcplot import plot_ppc
from .rankplot import plot_rank
from .traceplot import plot_trace
from .violinplot import plot_violin
