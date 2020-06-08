# pylint: disable=wrong-import-position
"""Bokeh Plotting Backend."""
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


from .autocorrplot import plot_autocorr
from .compareplot import plot_compare
from .densityplot import plot_density
from .distplot import plot_dist
from .elpdplot import plot_elpd
from .energyplot import plot_energy
from .essplot import plot_ess
from .forestplot import plot_forest
from .hdiplot import plot_hdi
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


def check_bokeh_version():
    """Check minimum bokeh version."""
    try:
        import bokeh

        assert version.parse(bokeh.__version__) >= version.parse("1.4.0")
    except (ImportError, AssertionError):
        raise ImportError("'bokeh' backend needs Bokeh (1.4.0+) installed.")
