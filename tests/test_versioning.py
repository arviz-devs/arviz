import pytest

from arviz._versioning import import_arviz_subpackage


def test_import_existing_module_with_version():
    # arviz itself always exists during tests
    version = import_arviz_subpackage("arviz")
    assert isinstance(version, str)


def test_import_missing_module_raises():
    with pytest.raises(
        ImportError,
        match="arviz's dependency does_not_exist is not installed",
    ):
        import_arviz_subpackage("does_not_exist")


def test_import_version_fallback(monkeypatch):
    class DummyModule:
        pass

    monkeypatch.setitem(__import__("sys").modules, "dummy_mod", DummyModule())

    version = import_arviz_subpackage("dummy_mod", version_fallback="0.7.0")
    assert version == "0.7.0"
