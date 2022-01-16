# pylint: disable=redefined-outer-name, no-member
"""Tests for arviz.utils."""
import importlib
from unittest.mock import Mock

import numpy as np
import pytest

from ...stats.stats_utils import stats_variance_2d as svar
from ...utils import Numba, _numba_var, numba_check
from ..helpers import running_on_ci
from .test_utils import utils_with_numba_import_fail  # pylint: disable=unused-import

pytestmark = pytest.mark.skipif(  # pylint: disable=invalid-name
    (importlib.util.find_spec("numba") is None) and not running_on_ci(),
    reason="test requires numba which is not installed",
)


def test_utils_fixture(utils_with_numba_import_fail):
    """Test of utils fixture to ensure mock is applied correctly"""

    # If Numba doesn't exist in dev environment this will raise an ImportError
    import numba  # pylint: disable=unused-import,W0612

    with pytest.raises(ImportError):
        utils_with_numba_import_fail.importlib.import_module("numba")


def test_conditional_jit_numba_decorator_keyword(monkeypatch):
    """Checks else statement and JIT keyword argument"""
    from ... import utils

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
