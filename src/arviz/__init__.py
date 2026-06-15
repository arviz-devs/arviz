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

# pyrefly: ignore [missing-import]
from xarray import open_datatree as from_netcdf

from_zarr = functools.partial(from_netcdf, engine="zarr")

_log = logging.getLogger(__name__) 

info = ""

_MIGRATION_GUIDE_URL = "https://python.arviz.org/en/latest/user_guide/migration_guide.html#datatree"


class MigrationWarning(UserWarning, FutureWarning):
    """Warning raised when a legacy name is accessed on the ``arviz`` namespace."""


try:
    # pyrefly: ignore [missing-import]
    from arviz_base import *
    # pyrefly: ignore [missing-import]
    import arviz_base as base

    _base_status = (
        f"arviz_base {base.__version__} available, "
        "exposing its functions as part of the `arviz` namespace"
    )
    _log.info(_base_status)
    del base
except ModuleNotFoundError as err:
    raise ImportError("arviz's dependency arviz_base is not installed", name="arviz") from err

info += _base_status + "\n"

try:
    # pyrefly: ignore [missing-import]
    from arviz_stats import *
    # pyrefly: ignore [missing-import]
    import arviz_stats as stats

    _stats_status = (
        f"arviz_stats {getattr(stats, '__version__', '0.7.0')} available, "
        "exposing its functions as part of the `arviz` namespace"
    )
    _log.info(_stats_status)
    del stats
except ModuleNotFoundError as err:
    raise ImportError("arviz's dependency arviz_stats is not installed", name="arviz") from err

info += _stats_status + "\n"

try:
    # pyrefly: ignore [missing-import]
    from arviz_plots import *
    # pyrefly: ignore [missing-import]
    import arviz_plots as plots

    _plots_status = (
        f"arviz_plots {plots.__version__} available, "
        "exposing its functions as part of the `arviz` namespace"
    )
    _log.info(_plots_status)
    del plots
except ModuleNotFoundError as err:
    raise ImportError("arviz's dependency arviz_plots is not installed", name="arviz") from err

info += _plots_status

# define version last so it isn't overwritten by the respective attribute in the imported libraries
__version__ = "1.2.0"

info = f"Status information for ArviZ {__version__}\n\n{info}"

pat = re.compile(r"arviz_(base|stats|plots)\s([0-9]+\.[0-9]+)")
versions = dict(pat.findall(info))
unique_versions = set(versions.values())

if len(unique_versions) > 1:
    lines = ["Incompatible ArviZ packages versions detected:"]

    for package, version in sorted(versions.items()):
        lines.append(f"- arviz_{package}: {version}")

    lines.append("All ArviZ packages must share the same minor version.")

    raise ImportError("\n".join(lines))

# ---------------------------------------------------------------------------
# Transition aliases — ease migration from pre-1.0 to 1.0
# These emit a MigrationWarning on first access and can be removed in a
# future major version once the ecosystem has fully migrated.
# ---------------------------------------------------------------------------

# plot_trace -> plot_trace_dist alias (defined eagerly so it shows up in dir())
try:
    import warnings as _warnings
    _plot_trace_dist = plot_trace_dist # noqa: F821 — exported by arviz_plots via *

    def plot_trace(*args, **kwargs):
        """Deprecated alias for :func:`plot_trace_dist`.

        .. deprecated::
            Use ``arviz.plot_trace_dist`` instead.
            See the migration guide: {url}
        """.format(url=_MIGRATION_GUIDE_URL)
        _warnings.warn(
            "arviz.plot_trace has been renamed to arviz.plot_trace_dist. "
            f"See the migration guide: {_MIGRATION_GUIDE_URL}",
            MigrationWarning,
            stacklevel=2,
        )
        return _plot_trace_dist(*args, **kwargs)

    del _warnings, _plot_trace_dist
except NameError:
    # plot_trace_dist not available in this version of arviz_plots — skip alias
    _log.debug("plot_trace_dist not found; skipping plot_trace alias")


def __getattr__(name):
    """Guide users who access legacy names on the ``arviz`` namespace."""
    import warnings
    # pyrefly: ignore [missing-import]
    from xarray import DataTree

    if name == "InferenceData":
        warnings.warn(
            "arviz.InferenceData is no longer available on the arviz package. "
            "ArviZ now uses xarray.DataTree for the same role. "
            f"See the migration guide: {_MIGRATION_GUIDE_URL}",
            MigrationWarning,
            stacklevel=2,  # points warning at the caller, not this file
        )
        return DataTree

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# clean namespace — only delete names that are guaranteed to exist
del functools, logging, pat, re, versions, unique_versions
del _base_status, _stats_status, _plots_status