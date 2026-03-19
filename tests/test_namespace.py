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
import importlib
import re

import arviz_base
import arviz_plots
import arviz_stats
import pytest

import arviz as az


def test_info_attr():
    info_message = az.info
    assert isinstance(info_message, str)

    pat = re.compile(r"arviz_(base|stats|plots)[\s\.0-9abdevrc]+available")
    for line in info_message.splitlines()[2:]:
        assert pat.match(line)


def test_aliases():
    # These are xarray aliases exposed for user convenience
    xarray_aliases = {"from_netcdf", "from_zarr"}

    for obj_name in dir(az):
        if not obj_name.startswith("_") and obj_name != "info":
            obj = getattr(az, obj_name)

            if obj_name in xarray_aliases:
                # from_netcdf and from_zarr are xarray.open_datatree aliases
                continue

            if hasattr(obj, "__module__"):
                orig_lib = obj.__module__.split(".")[0]
            elif hasattr(obj, "__package__"):
                orig_lib = obj.__package__
            else:
                pytest.fail(obj_name)

            assert orig_lib.startswith("arviz"), obj_name
            assert orig_lib != "arviz", obj_name


def test_from_netcdf_alias():
    """Test that from_netcdf is an alias for xarray.open_datatree."""
    import xarray

    assert az.from_netcdf is xarray.open_datatree


def test_from_zarr_alias():
    """Test that from_zarr is a partial of xarray.open_datatree with engine='zarr'."""
    import xarray

    assert az.from_zarr.func is xarray.open_datatree
    assert az.from_zarr.keywords == {"engine": "zarr"}


def test_incompatible_package_versions(monkeypatch):
    monkeypatch.setattr(arviz_base, "__version__", "0.7.0")
    monkeypatch.setattr(arviz_stats, "__version__", "0.6.0", raising=False)
    monkeypatch.setattr(arviz_plots, "__version__", "0.7.0")

    with pytest.raises(ImportError) as excinfo:
        importlib.reload(az)

    message = str(excinfo.value)

    assert "Incompatible ArviZ packages versions detected" in message
    assert "- arviz_base: 0.7" in message
    assert "- arviz_stats: 0.6" in message
    assert "- arviz_plots: 0.7" in message
    assert "must share the same minor version" in message


def test_inference_data_getattr_points_to_migration_guide():
    """Legacy arviz.InferenceData should error with a link to the migration guide."""
    with pytest.raises(ImportError) as excinfo:
        getattr(az, "InferenceData")

    assert "python.arviz.org" in str(excinfo.value)
    assert "migration_guide" in str(excinfo.value)


def test_getattr_unknown_attribute():
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(az, "totally_missing_arviz_name_xyz")


def test_compatible_package_versions(monkeypatch):
    """
    Compatible minor versions should not raise an ImportError and should be
    reported correctly.
    """

    # Same minor version (0.7.x) → should be allowed
    monkeypatch.setattr(arviz_base, "__version__", "0.7.0", raising=False)
    monkeypatch.setattr(arviz_stats, "__version__", "0.7.1", raising=False)
    monkeypatch.setattr(arviz_plots, "__version__", "0.7.2", raising=False)

    # Should not raise
    importlib.reload(az)

    # Assert info reports the patched versions correctly
    info = az.info

    assert "arviz_base 0.7.0 available" in info
    assert "arviz_stats 0.7.1 available" in info
    assert "arviz_plots 0.7.2 available" in info
