# pylint: disable=wrong-import-position
"""Bokeh Plotting Backend."""
from bokeh.plotting import figure
from numpy import array
from packaging import version

from ....rcparams import rcParams


def backend_kwarg_defaults(*args, **kwargs):
    """Get default kwargs for backend.

    For args add a tuple with key and rcParam key pair.
    """
    defaults = {**kwargs}
    # add needed default args from arviz.rcParams
    for key, arg in args:
        defaults.setdefault(key, rcParams[arg])

    for key, arg in {
        "toolbar_location": "plot.bokeh.layout.toolbar_location",
        "tools": "plot.bokeh.tools",
        "output_backend": "plot.bokeh.output_backend",
        "height": "plot.bokeh.figure.height",
        "width": "plot.bokeh.figure.width",
    }.items():
        # by default, ignore height and width if dpi is used
        if key in ("height", "width") and "dpi" in defaults:
            continue
        defaults.setdefault(key, rcParams[arg])
    return defaults


def create_axes_grid(
    length_plotters,
    rows=1,
    cols=1,
    figsize=None,
    squeeze=False,
    sharex=False,
    sharey=False,
    polar=False,
    backend_kwargs=None,
):
    """Create figure and axes for grids with multiple plots.

    Parameters
    ----------
    length_plotters : int
        Number of figures required
    rows : int
        Number of rows
    cols : int
        Number of columns
    figsize : tuple
        Figure size in inches
    squeeze : bool
        Return bokeh.figure object if True else ndarray of bokeh.figure objects
    sharex : bool
        Share x axis between the figures
    sharey : bool
        Share y axis between the figures
    polar : bool
        Set up polar coordinates plot
    backend_kwargs: dict, optional
        kwargs for backend figure.

    Returns
    -------
    figures : bokeh figure or np.array of bokeh.figure
    """
    if backend_kwargs is None:
        backend_kwargs = {}

    if figsize is not None:
        backend_kwargs = {
            **backend_kwarg_defaults(
                ("dpi", "plot.bokeh.figure.dpi"),
            ),
            **backend_kwargs,
        }
        dpi = backend_kwargs.pop("dpi")
        backend_kwargs.setdefault("width", int(figsize[0] * dpi / cols))
        backend_kwargs.setdefault("height", int(figsize[1] * dpi / rows))
    else:
        backend_kwargs = {
            **backend_kwarg_defaults(),
            **backend_kwargs,
        }

    figures = []

    if polar:
        backend_kwargs.setdefault("x_axis_type", None)
        backend_kwargs.setdefault("y_axis_type", None)

    for row in range(rows):
        row_figures = []
        for col in range(cols):
            if (row == 0) and (col == 0) and (sharex or sharey):
                p = figure(**backend_kwargs)  # pylint: disable=invalid-name
                row_figures.append(p)
                if sharex:
                    backend_kwargs["x_range"] = p.x_range
                if sharey:
                    backend_kwargs["y_range"] = p.y_range
            elif row * cols + (col + 1) > length_plotters:
                row_figures.append(None)
            else:
                row_figures.append(figure(**backend_kwargs))
        figures.append(row_figures)
    figures = array(figures)
    if figures.size == 1 and squeeze:
        figures = figures[0, 0]
    return figures


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
        **{prop: props[idx] for prop, props in prop_dict.items()},
    }


from .autocorrplot import plot_autocorr
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
from .ppcplot import plot_ppc
from .posteriorplot import plot_posterior
from .rankplot import plot_rank
from .traceplot import plot_trace
from .violinplot import plot_violin


def check_bokeh_version():
    """Check minimum bokeh version."""
    try:
        import bokeh

        assert version.parse(bokeh.__version__) >= version.parse("1.4.0")
    except (ImportError, AssertionError) as err:
        raise ImportError("'bokeh' backend needs Bokeh (1.4.0+) installed.") from err
