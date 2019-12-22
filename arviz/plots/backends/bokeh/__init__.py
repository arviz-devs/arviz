# pylint: disable=no-member,invalid-name,redefined-outer-name, wrong-import-position
"""Bokeh Plotting Backend."""
import packaging


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
        show = rcParams["plot.bokeh.show"]
    return show


from .autocorrplot import plot_autocorr
from .compareplot import plot_compare
from .densityplot import plot_density
from .distplot import plot_dist
from .elpdplot import plot_elpd
from .energyplot import plot_energy
from .essplot import plot_ess
from .forestplot import plot_forest
from .hpdplot import plot_hpd
from .jointplot import plot_joint
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
from ....rcparams import rcParams


def output_notebook(*args, **kwargs):
    """Wrap bokeh.plotting.output_notebook."""
    import bokeh.plotting as bkp

    return bkp.output_notebook(*args, **kwargs)


def output_file(*args, **kwargs):
    """Wrap bokeh.plotting.output_file."""
    import bokeh.plotting as bkp

    return bkp.output_file(*args, **kwargs)


def copy_docstring(lib, function):
    """Extract docstring from function."""
    import importlib

    try:
        module = importlib.import_module(lib)
        func = getattr(module, function)
        doc = func.__doc__
    except ImportError:
        doc = "Failed to import {}".format(lib)

    return doc


def check_bokeh_version():
    """Check minimum bokeh version."""
    try:
        import bokeh

        assert packaging.version.parse(bokeh.__version__) >= packaging.version.parse("1.4.0")
    except (ImportError, AssertionError):
        raise ImportError("'bokeh' backend needs Bokeh (1.4.0+) installed.")


def ColumnDataSource(*args, **kwargs):
    """Wrap bokeh.models.ColumnDataSource."""
    from bokeh.models import ColumnDataSource

    return ColumnDataSource(*args, **kwargs)


output_notebook.__doc__ += "\n\n" + copy_docstring("bokeh.plotting", "output_notebook")
output_file.__doc__ += "\n\n" + copy_docstring("bokeh.plotting", "output_file")
ColumnDataSource.__doc__ += "\n\n" + copy_docstring("bokeh.models", "ColumnDataSource")
