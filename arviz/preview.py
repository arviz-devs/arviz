# pylint: disable=unused-import,unused-wildcard-import,wildcard-import,invalid-name
"""Expose features from arviz-xyz refactored packages inside ``arviz.preview`` namespace."""
import logging

_log = logging.getLogger(__name__)

info = ""

try:
    from arviz_base import *

    status = "arviz_base available, exposing its functions as part of arviz.preview"
    _log.info(status)
except ModuleNotFoundError:
    status = "arviz_base not installed"
    _log.info(status)
except ImportError:
    status = "Unable to import arviz_base"
    _log.info(status, exc_info=True)

info += status + "\n"

try:
    from arviz_stats import *

    status = "arviz_stats available, exposing its functions as part of arviz.preview"
    _log.info(status)
except ModuleNotFoundError:
    status = "arviz_stats not installed"
    _log.info(status)
except ImportError:
    status = "Unable to import arviz_stats"
    _log.info(status, exc_info=True)
info += status + "\n"

try:
    from arviz_plots import *

    status = "arviz_plots available, exposing its functions as part of arviz.preview"
    _log.info(status)
except ModuleNotFoundError:
    status = "arviz_plots not installed"
    _log.info(status)
except ImportError:
    status = "Unable to import arviz_plots"
    _log.info(status, exc_info=True)

info += status + "\n"
