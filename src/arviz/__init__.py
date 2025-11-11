# pylint: disable=unused-import,unused-wildcard-import,wildcard-import,invalid-name
"""Expose features from _ArviZverse_ refactored packages together in the ``arviz`` namespace."""

import logging

_log = logging.getLogger(__name__)

info = ""

try:
    from arviz_base import *
    import arviz_base as base

    _status = (
        f"arviz_base {base.__version__} available, "
        "exposing its functions as part of the `arviz` namespace"
    )
    _log.info(_status)
except ModuleNotFoundError:
    _status = "arviz_base not installed"
    _log.exception(_status)
except ImportError:
    _status = "Unable to import arviz_base"
    _log.exception(_status)

info += _status + "\n"

try:
    from arviz_stats import *

    # the base computational module fron arviz_stats will override the alias to arviz-base
    # arviz.stats.base will still be available
    import arviz_base as base
    import arviz_stats as stats

    _status = (
        f"arviz_stats {getattr(stats, '__version__', '0.0')} available, "
        "exposing its functions as part of the `arviz` namespace"
    )
    _log.info(_status)
except ModuleNotFoundError:
    _status = "arviz_stats not installed"
    _log.exception(_status)
except ImportError:
    _status = "Unable to import arviz_stats"
    _log.exception(_status)
info += _status + "\n"

try:
    from arviz_plots import *
    import arviz_plots as plots

    _status = (
        f"arviz_plots {plots.__version__} available, "
        "exposing its functions as part of the `arviz` namespace"
    )
    _log.info(_status)
except ModuleNotFoundError:
    _status = "arviz_plots not installed"
    _log.exception(_status)
except ImportError:
    _status = "Unable to import arviz_plots"
    _log.exception(_status)

info += _status

# define version last so it isn't overwritten by the respective attribute in the imported libraries
__version__ = "1.0.0.dev0"

info = f"Status information for ArviZ {__version__}\n\n{info}"

# clean namespace
del logging, _status
