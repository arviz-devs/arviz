"""
Tests for arviz.utils.
"""
# pylint: disable=redefined-outer-name, no-member
from unittest.mock import Mock
import importlib
import numpy as np
import pytest

from ..utils import _var_names, numba_check, Numba, _numba_var, _stack, one_de, two_de, expand_dims
from ..data import load_arviz_data, from_dict
from ..stats.stats_utils import stats_variance_2d as svar


@pytest.fixture(scope="session")
def data():
    centered_eight = load_arviz_data("centered_eight")
    return centered_eight.posterior


@pytest.mark.parametrize(
    "var_names_expected",
    [
        ("mu", ["mu"]),
        (None, None),
        (["mu", "tau"], ["mu", "tau"]),
        ("~mu", ["theta", "tau"]),
        (["~mu"], ["theta", "tau"]),
    ],
)
def test_var_names(var_names_expected, data):
    """Test var_name handling"""
    var_names, expected = var_names_expected
    assert _var_names(var_names, data) == expected


def test_var_names_warning():
    """Test confusing var_name handling"""
    data = from_dict(
        posterior={
            "~mu": np.random.randn(2, 10),
            "mu": -np.random.randn(2, 10),  # pylint: disable=invalid-unary-operand-type
            "theta": np.random.randn(2, 10, 8),
        }
    ).posterior
    var_names = expected = ["~mu"]
    with pytest.warns(UserWarning):
        assert _var_names(var_names, data) == expected


def test_var_names_key_error(data):
    with pytest.raises(KeyError, match="bad_var_name"):
        _var_names(("theta", "tau", "bad_var_name"), data)


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


def test_conditional_vect_decorator_no_numba(utils_with_numba_import_fail):
    """Tests to see if Numba vectorize code block is skipped with Import Failure

    Test can be distinguished from test_conditional_vect__numba_decorator
    by use of debugger or coverage tool
    """

    @utils_with_numba_import_fail.conditional_vect
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
        return True

    assert func()


def test_conditional_jit_numba_decorator_keyword(monkeypatch):
    """Checks else statement and JIT keyword argument"""
    from arviz import utils

    # Mock import lib to return numba with hit method which returns a function that returns kwargs
    numba_mock = Mock()
    monkeypatch.setattr(utils.importlib, "import_module", lambda x: numba_mock)

    def jit(**kwargs):
        """overwrite numba.jit function"""
        return lambda fn: lambda: (fn(), kwargs)

    numba_mock.jit = jit

    @utils.conditional_jit(keyword_argument="A keyword argument")
    def placeholder_func():
        """This function does nothing"""
        return "output"

    # pylint: disable=unpacking-non-sequence
    function_results, wrapper_result = placeholder_func()
    assert wrapper_result == {"keyword_argument": "A keyword argument"}
    assert function_results == "output"


def test_conditional_vect_numba_decorator():
    """Tests to see if Numba is used.

    Test can be distinguished from test_conditional_jit_decorator_no_numba
    by use of debugger or coverage tool
    """
    from arviz import utils

    @utils.conditional_vect
    def func(a_a, b_b):
        return a_a + b_b

    value_one = np.random.randn(10)
    value_two = np.random.randn(10)
    assert np.allclose(func(value_one, value_two), value_one + value_two)


def test_conditional_vect_numba_decorator_keyword(monkeypatch):
    """Checks else statement and vect keyword argument"""
    from arviz import utils

    # Mock import lib to return numba with hit method which returns a function that returns kwargs
    numba_mock = Mock()
    monkeypatch.setattr(utils.importlib, "import_module", lambda x: numba_mock)

    def vectorize(**kwargs):
        """overwrite numba.vectorize function"""
        return lambda x: (x(), kwargs)

    numba_mock.vectorize = vectorize

    @utils.conditional_vect(keyword_argument="A keyword argument")
    def placeholder_func():
        """This function does nothing"""
        return "output"

    # pylint: disable=unpacking-non-sequence
    function_results, wrapper_result = placeholder_func
    assert wrapper_result == {"keyword_argument": "A keyword argument"}
    assert function_results == "output"


def test_numba_check():
    """Test for numba_check"""
    numba = importlib.util.find_spec("numba")
    flag = numba is not None
    assert flag == numba_check()


def test_numba_utils():
    """Test for class Numba."""
    flag = Numba.numba_flag
    assert flag == numba_check()
    Numba.disable_numba()
    val = Numba.numba_flag
    assert not val
    Numba.enable_numba()
    val = Numba.numba_flag
    assert val
    assert flag == Numba.numba_flag


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("ddof", (0, 1))
def test_numba_var(axis, ddof):
    """Method to test numba_var."""
    flag = Numba.numba_flag
    data_1 = np.random.randn(100, 100)
    data_2 = np.random.rand(100)
    with_numba_1 = _numba_var(svar, np.var, data_1, axis=axis, ddof=ddof)
    with_numba_2 = _numba_var(svar, np.var, data_2, ddof=ddof)
    Numba.disable_numba()
    non_numba_1 = _numba_var(svar, np.var, data_1, axis=axis, ddof=ddof)
    non_numba_2 = _numba_var(svar, np.var, data_2, ddof=ddof)
    Numba.enable_numba()
    assert flag == Numba.numba_flag
    assert np.allclose(with_numba_1, non_numba_1)
    assert np.allclose(with_numba_2, non_numba_2)


def test_stack():
    x = np.random.randn(10, 4, 6)
    y = np.random.randn(100, 4, 6)
    assert x.shape[1:] == y.shape[1:]
    assert np.allclose(np.vstack((x, y)), _stack(x, y))
    assert _stack


@pytest.mark.parametrize("data", [np.random.randn(1000), np.random.randn(1000).tolist()])
def test_two_de(data):
    """Test to check for custom atleast_2d. List added to test for a non ndarray case."""
    assert np.allclose(two_de(data), np.atleast_2d(data))


@pytest.mark.parametrize("data", [np.random.randn(100), np.random.randn(100).tolist()])
def test_one_de(data):
    """Test to check for custom atleast_1d. List added to test for a non ndarray case."""
    assert np.allclose(one_de(data), np.atleast_1d(data))


@pytest.mark.parametrize("data", [np.random.randn(100), np.random.randn(100).tolist()])
def test_expand_dims(data):
    """Test to check for custom expand_dims. List added to test for a non ndarray case."""
    assert np.allclose(expand_dims(data), np.expand_dims(data, 0))
