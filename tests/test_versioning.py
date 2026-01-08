import pytest

from arviz._versioning import import_arviz_subpackage


def test_import_existing_module():
    version = import_arviz_subpackage("arviz")
    assert isinstance(version, str)


def test_missing_module():
    with pytest.raises(ImportError):
        import_arviz_subpackage("does_not_exist")


def test_version_fallback(monkeypatch):
    class Dummy:
        pass

    monkeypatch.setitem(__import__("sys").modules, "dummy_mod", Dummy())
    assert import_arviz_subpackage("dummy_mod", version_fallback="0.7.0") == "0.7.0"
