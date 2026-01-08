# pylint: disable=unused-import,unused-wildcard-import,wildcard-import,invalid-name
"""Expose features from _ArviZverse_ refactored packages together in the ``arviz`` namespace."""

import functools
import logging
import re
from ._versioning import import_arviz_subpackage
from xarray import open_datatree as from_netcdf

from_zarr = functools.partial(from_netcdf, engine="zarr")

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

from arviz_stats import *

version = import_arviz_subpackage("arviz_stats", version_fallback="0.7.0")

_status = (
    f"arviz_stats {version} available, exposing its functions as part of the `arviz` namespace"
)

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

versions = dict(pat.findall(info))
unique_versions = set(versions.values())

if len(unique_versions) > 1:
    lines = ["Incompatible ArviZ subpackage versions detected:"]

    for package, version in sorted(versions.items()):
        lines.append(f"- arviz_{package}: {version}")

    lines.append("All ArviZ subpackages must share the same minor version.")

    raise ImportError("\n".join(lines))


# clean namespace
del (
    functools,
    import_arviz_subpackage,
    version,
    logging,
    matches,
    pat,
    re,
    _status,
    versions,
    unique_versions,
)
