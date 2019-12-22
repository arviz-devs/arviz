# pylint: disable= wrong-import-position
"""Matplotlib Plotting Backend."""


def backend_kwarg_defaults(*args, **kwargs):
    """Get default kwargs for backend.

    For args add a tuple with key and rcParam key pair.
    """
    defaults = {**kwargs}
    # add needed default args from arviz.rcParams
    for key, arg in args:
        defaults.setdefault(key, rcParams[arg])
    # add default args from rcParams
    defaults.setdefault("constrained_layout", rcParams["plot.matplotlib.constrained_layout"])
    return defaults


def backend_show(show):
    """Set default behaviour for show if not explicitly defined."""
    if show is None:
        show = rcParams["plot.matplotlib.show"]
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
from .posteriorplot import plot_posterior
from .ppcplot import plot_ppc
from .rankplot import plot_rank
from .traceplot import plot_trace
from .violinplot import plot_violin
from ....rcparams import rcParams
