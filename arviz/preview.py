# pylint: disable=unused-import,unused-wildcard-import,wildcard-import,invalid-name
"""Expose features from arviz-xyz refactored packages inside ``arviz.preview`` namespace."""
import logging

_log = logging.getLogger(__name__)

info = ""

try:
    from arviz_base import *
    import arviz_base as base

    _status = "arviz_base available, exposing its functions as part of arviz.preview"
    _log.info(_status)
except ModuleNotFoundError:
    _status = "arviz_base not installed"
    _log.info(_status)
except ImportError:
    _status = "Unable to import arviz_base"
    _log.info(_status, exc_info=True)

info += _status + "\n"

try:
    from arviz_stats import *

    # the base computational module fron arviz_stats will override the alias to arviz-base
    # arviz.stats.base will still be available
    import arviz_base as base
    import arviz_stats as stats

    _status = "arviz_stats available, exposing its functions as part of arviz.preview"
    _log.info(_status)
except ModuleNotFoundError:
    _status = "arviz_stats not installed"
    _log.info(_status)
except ImportError:
    _status = "Unable to import arviz_stats"
    _log.info(_status, exc_info=True)
info += _status + "\n"

try:
    from arviz_plots import *
    import arviz_plots as plots

    _status = "arviz_plots available, exposing its functions as part of arviz.preview"
    _log.info(_status)
except ModuleNotFoundError:
    _status = "arviz_plots not installed"
    _log.info(_status)
except ImportError:
    _status = "Unable to import arviz_plots"
    _log.info(_status, exc_info=True)

info += _status + "\n"

# clean namespace
del logging, _status, _log
