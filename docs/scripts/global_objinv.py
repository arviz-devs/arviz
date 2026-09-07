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
"""Script to extract entries from arvizverse and create a global objects.inv.

Having a global objects.inv allows downstream docs and examples to use
:func:`arviz.plot_dist` or :func:`arviz.ess` and get the correct cross-references
pointing to base/stats/plots respective API page.
"""

import os
from pathlib import Path

import sphobjinv as soi


def update_inv_entries(partial_inv, target_version="stable"):
    """Extract a list of sphobjinv objects from the partial inventory and update it.

    Updates make the object reflect the alternative/recommended import from the global
    arviz namespace.
    """
    partial_modname = partial_inv.project.replace("-", "_")
    partial_alias = partial_modname.split("_")[1]
    version_str = "latest" if target_version == "dev" else target_version
    updated_entries = [
        entry.evolve(
            name=entry.name.replace(partial_modname, "arviz"),
            uri=f"projects/{partial_alias}/en/{version_str}/{entry.uri}",
        )
        for entry in partial_inv.objects
        if (
            (entry.domain == "py")
            and (entry.name != partial_modname)
            and (not entry.name.startswith("arviz_plots.backend"))
            and (not entry.name.startswith("arviz_stats.base"))
            and (not entry.name.startswith("arviz_stats.numba"))
        )
    ]
    return updated_entries


if __name__ == "__main__":
    version = os.environ.get("READTHEDOCS_VERSION", "stable")
    if version != "latest":
        version = "stable"
    output_dir = Path(os.environ.get("READTHEDOCS_OUTPUT", "docs/build"))

    base_inv = soi.Inventory(url=f"https://python.arviz.org/projects/base/en/{version}/objects.inv")
    stats_inv = soi.Inventory(
        url=f"https://python.arviz.org/projects/stats/en/{version}/objects.inv"
    )
    plots_inv = soi.Inventory(
        url=f"https://python.arviz.org/projects/plots/en/{version}/objects.inv"
    )

    arviz_inv = soi.Inventory(output_dir / "html" / "objects.inv")
    for entry in arviz_inv.objects:
        entry.uri = f"en/{version}/{entry.uri}"

    arviz_inv.objects.extend(update_inv_entries(base_inv, version))
    arviz_inv.objects.extend(update_inv_entries(stats_inv, version))
    arviz_inv.objects.extend(update_inv_entries(plots_inv, version))

    df = arviz_inv.data_file()
    dfc = soi.compress(df)
    soi.writebytes(output_dir / "html" / "objects.inv", dfc)
