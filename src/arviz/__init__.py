# ﻿Copyright ArviZ contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=unused-import,unused-wildcard-import,wildcard-import,invalid-name
"""Expose features from _ArviZverse_ refactored packages together in the ``arviz`` namespace."""

import functools
import logging
import re

from xarray import open_datatree as from_netcdf

from_zarr = functools.partial(from_netcdf, engine="zarr")

_log = logging.getLogger(__name__)

info = ""


class MigrationWarning(FutureWarning):
    """Warning raised when a legacy name is accessed on the ``arviz`` namespace."""


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
__version__ = "1.0.0"

info = f"Status information for ArviZ {__version__}\n\n{info}"

pat = re.compile(r"arviz_(base|stats|plots)\s([0-9]+\.[0-9]+)")
matches = pat.findall(info)

versions = dict(pat.findall(info))
unique_versions = set(versions.values())

if len(unique_versions) > 1:
    lines = ["Incompatible ArviZ packages versions detected:"]

    for package, version in sorted(versions.items()):
        lines.append(f"- arviz_{package}: {version}")

    lines.append("All ArviZ packages must share the same minor version.")

    raise ImportError("\n".join(lines))

_MIGRATION_GUIDE_URL = "https://python.arviz.org/en/latest/user_guide/migration_guide.html#datatree"


def __getattr__(name):
    """Guide users who expect legacy names on the ``arviz`` namespace."""
    if name == "InferenceData":
        import warnings
        from xarray import DataTree

        warnings.warn(
            "arviz.InferenceData is no longer available on the "
            "arviz package; ArviZ now uses xarray's DataTree for the same "
            f"role. See the migration guide: {_MIGRATION_GUIDE_URL}",
            MigrationWarning,
        )

        return DataTree
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# clean namespace
del functools, logging, matches, pat, re, _status, versions, unique_versions
