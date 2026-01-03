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
    for obj_name in dir(az):
        if not obj_name.startswith("_") and obj_name != "info":
            obj = getattr(az, obj_name)

            if hasattr(obj, "__module__"):
                orig_lib = obj.__module__.split(".")[0]
            elif hasattr(obj, "__package__"):
                orig_lib = obj.__package__
            else:
                pytest.fail(obj_name)

            assert orig_lib.startswith("arviz"), obj_name
            assert orig_lib != "arviz", obj_name


def test_incompatible_subpackage_versions(monkeypatch):
    monkeypatch.setattr(arviz_base, "__version__", "0.7.0")
    monkeypatch.setattr(arviz_stats, "__version__", "0.6.0", raising=False)
    monkeypatch.setattr(arviz_plots, "__version__", "0.7.0")

    with pytest.raises(ImportError) as excinfo:
        importlib.reload(az)

    message = str(excinfo.value)

    assert "Incompatible ArviZ subpackage versions detected" in message
    assert "- arviz_base: 0.7" in message
    assert "- arviz_stats: 0.6" in message
    assert "- arviz_plots: 0.7" in message
    assert "must share the same minor version" in message


def test_compatible_subpackage_versions(monkeypatch):
    """Compatible minor versions should not raise an ImportError."""

    # Same minor version (0.7.x) → should be allowed
    monkeypatch.setattr(arviz_base, "__version__", "0.7.0", raising=False)
    monkeypatch.setattr(arviz_stats, "__version__", "0.7.1", raising=False)
    monkeypatch.setattr(arviz_plots, "__version__", "0.7.2", raising=False)

    # Should not raise
    importlib.reload(az)
