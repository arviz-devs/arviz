"""
Tests for arviz.utils.
"""
# pylint: disable=redefined-outer-name
from unittest.mock import Mock
import pytest
from ..utils import _var_names


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

    monkeypatch.setattr(utils.importlib, "import_module", failed_import)
    return utils


def test_utils_fixture(utils_with_numba_import_fail):
    """Test of utils fixture to ensure mock is applied correctly"""

    # If Numba doesn't exist in dev environment this will raise an ImportError
    import numba  # pylint: disable=unused-import,W0612

    with pytest.raises(ImportError):
        utils_with_numba_import_fail.importlib.import_module("numba")


def test_conditional_jit_decorator_no_numba(utils_with_numba_import_fail):
    """Tests to see if Numba jit code block is skipped with Import Failure

    Test can be distinguished from test_conditional_jit__numba_decorator
    by use of debugger or coverage tool
    """

    @utils_with_numba_import_fail.conditional_jit
    def func():
        return "Numba not used"

    assert func() == "Numba not used"


def test_conditional_jit_numba_decorator():
    """Tests to see if Numba is used.

    Test can be distinguished from test_conditional_jit_decorator_no_numba
    by use of debugger or coverage tool
    """
    from arviz import utils

    @utils.conditional_jit
    def func():
        return "Numba used"

    assert func() == "Numba used"


def test_conditional_jit_numba_decorator_keyword(monkeypatch):
    """Checks else statement and JIT keyword argument"""
    from arviz import utils

    # Mock import lib to return numba with hit method which returns a function that returns kwargs
    numba_mock = Mock()
    monkeypatch.setattr(utils.importlib, "import_module", lambda x: numba_mock)

    def jit(**kwargs):
        """overwrite numba.jit function"""
        return lambda x: (x(), kwargs)

    numba_mock.jit = jit

    @utils.conditional_jit(keyword_argument="A keyword argument")
    def placeholder_func():
        """This function does nothing"""
        return "output"

    # pylint: disable=unpacking-non-sequence
    function_results, wrapper_result = placeholder_func
    assert wrapper_result == {"keyword_argument": "A keyword argument"}
    assert function_results == "output"
