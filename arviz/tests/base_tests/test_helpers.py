import pytest
from _pytest.outcomes import Skipped

from ..helpers import importorskip


def test_importorskip_local(monkeypatch):
    """Test ``importorskip`` run on local machine with non-existent module, which should skip."""
    monkeypatch.delenv("ARVIZ_REQUIRE_ALL_DEPS", raising=False)
    with pytest.raises(Skipped):
        importorskip("non-existent-function")


def test_importorskip_ci(monkeypatch):
    """Test ``importorskip`` run on CI machine with non-existent module, which should fail."""
    monkeypatch.setenv("ARVIZ_REQUIRE_ALL_DEPS", 1)
    with pytest.raises(ModuleNotFoundError):
        importorskip("non-existent-function")
