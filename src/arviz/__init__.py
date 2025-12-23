# pylint: disable=unused-import,unused-wildcard-import,wildcard-import,invalid-name
"""Expose features from _ArviZverse_ refactored packages together in the ``arviz`` namespace."""

import logging
import re

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
    del base
except ModuleNotFoundError as err:
    raise ImportError("arviz's dependency arviz_base is not installed", name="arviz") from err

info += _status + "\n"

try:
    from arviz_stats import *
    import arviz_stats as stats

    # TODO: remove patch. 0.7 version of arviz-stats didn't expose the __version__ attribute
    _status = (
        f"arviz_stats {getattr(stats, '__version__', '0.7.0')} available, "
        "exposing its functions as part of the `arviz` namespace"
    )
    _log.info(_status)
    del stats
except ModuleNotFoundError as err:
    raise ImportError("arviz's dependency arviz_stats is not installed", name="arviz") from err

info += _status + "\n"

try:
    from arviz_plots import *
    import arviz_plots as plots

    _status = (
        f"arviz_plots {plots.__version__} available, "
        "exposing its functions as part of the `arviz` namespace"
    )
    _log.info(_status)
    del plots
except ModuleNotFoundError as err:
    raise ImportError("arviz's dependency arviz_plots is not installed", name="arviz") from err

info += _status

# define version last so it isn't overwritten by the respective attribute in the imported libraries
__version__ = "1.0.0rc0"

info = f"Status information for ArviZ {__version__}\n\n{info}"

pat = re.compile(r"arviz_(base|stats|plots)\s([0-9]+\.[0-9]+)")
matches = pat.findall(info)

versions = {name: version for name, version in matches}
unique_versions = set(versions.values())

if len(unique_versions) > 1:
    raise ImportError(
        "Incompatible ArviZ subpackage versions detected. "
        "All arviz-* packages must share the same minor version. "
        f"Found versions: {versions}"
    )

# clean namespace
del logging, matches, pat, re, _status, versions, unique_versions
