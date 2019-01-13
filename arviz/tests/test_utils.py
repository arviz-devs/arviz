"""
Tests for arviz.utils
"""
import pytest
from unittest.mock import Mock
from arviz.utils import _var_names
from arviz import utils


@pytest.mark.parametrize(
    "var_names_expected", [("mu", ["mu"]), (None, None), (["mu", "tau"], ["mu", "tau"])]
)
def test_var_names(var_names_expected):
    """Test var_name handling"""
    var_names, expected = var_names_expected
    assert _var_names(var_names) == expected


@pytest.fixture(scope="function")
def utils_with_numba_import_fail(monkeypatch):
    """Patch numba in utils so when its imported it raises ImportError"""
    failed_import = Mock()
    failed_import.side_effect = ImportError

    from arviz import utils
    monkeypatch.setattr(utils.importlib, 'import_module', failed_import)
    return utils


def test_utils_fixture(utils_with_numba_import_fail):
    """Meta test of utils fixture to ensure things are mocked out correctly"""
    with pytest.raises(ImportError):
        utils.importlib.import_module("numba")


def test_conditional_jit_decorator_no_numba(utils_with_numba_import_fail):
    """Tests to see if Numba jit code block is skipped with Import Failure

    Test can be distinguished from test_conditional_jit__numba_decorator
    by use of debugger or coverage tool
    """
    utils = utils_with_numba_import_fail

    @utils.conditional_jit
    def func():
        return "Numba not used"

    assert "Numba not used" == func()


def test_conditional_jit__numba_decorator():
    """Tests to see if Numba is used when import failure.

    Test can be distinguished from test_conditional_jit_decorator_no_numba
    by use of debugger or coverage tool
    """
    @utils.conditional_jit
    def func():
        return "Numba used"

    assert "Numba used" == func()

