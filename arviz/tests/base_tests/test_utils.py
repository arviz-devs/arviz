"""
Tests for arviz.utils.
"""
# pylint: disable=redefined-outer-name, no-member
from unittest.mock import Mock
import numpy as np
import pytest

from arviz.data.base import dict_to_dataset
from ...utils import (
    _var_names,
    _stack,
    one_de,
    two_de,
    expand_dims,
    flatten_inference_data_to_dict,
    _subset_list,
)
from ...data import load_arviz_data, from_dict


@pytest.fixture(scope="session")
def inference_data():
    centered_eight = load_arviz_data("centered_eight")
    return centered_eight


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


@pytest.mark.parametrize(
    "var_args",
    [
        (["alpha", "beta"], ["alpha", "beta1", "beta2"], "like"),
        (["~beta"], ["alpha", "p1", "p2", "phi", "theta", "theta_t"], "like"),
        (["theta"], ["theta", "theta_t"], "like"),
        (["~theta"], ["alpha", "beta1", "beta2", "p1", "p2", "phi"], "like"),
        (["p"], ["alpha", "p1", "p2", "phi"], "like"),
        (["~p"], ["beta1", "beta2", "theta", "theta_t"], "like"),
        (["^bet"], ["beta1", "beta2"], "regex"),
        (["^p"], ["p1", "p2", "phi"], "regex"),
        (["~^p"], ["alpha", "beta1", "beta2", "theta", "theta_t"], "regex"),
        (["p[0-9]+"], ["p1", "p2"], "regex"),
        (["~p[0-9]+"], ["alpha", "beta1", "beta2", "phi", "theta", "theta_t"], "regex"),
    ],
)
def test_var_names_filter(var_args):
    """Test var_names filter with partial naming or regular expressions."""
    samples = np.random.randn(10)
    data = dict_to_dataset(
        {
            "alpha": samples,
            "beta1": samples,
            "beta2": samples,
            "p1": samples,
            "p2": samples,
            "phi": samples,
            "theta": samples,
            "theta_t": samples,
        }
    )
    var_names, expected, filter_vars = var_args
    assert _var_names(var_names, data, filter_vars) == expected


def test_subset_list_negation_not_found():
    """Check there is a warning if negation pattern is ignored"""
    names = ["mu", "theta"]
    with pytest.warns(UserWarning, match=".+not.+found.+"):
        assert _subset_list("~tau", names) == names


@pytest.fixture(scope="function")
def utils_with_numba_import_fail(monkeypatch):
    """Patch numba in utils so when its imported it raises ImportError"""
    failed_import = Mock()
    failed_import.side_effect = ImportError

    from arviz import utils

    monkeypatch.setattr(utils.importlib, "import_module", failed_import)
    return utils


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


@pytest.mark.parametrize("var_names", [None, "mu", ["mu", "tau"]])
@pytest.mark.parametrize(
    "groups", [None, "posterior_groups", "prior_groups", ["posterior", "sample_stats"]]
)
@pytest.mark.parametrize("dimensions", [None, "draw", ["chain", "draw"]])
@pytest.mark.parametrize("group_info", [True, False])
@pytest.mark.parametrize(
    "var_name_format", [None, "brackets", "underscore", "cds", ((",", "[", "]"), ("_", ""))]
)
@pytest.mark.parametrize("index_origin", [None, 0, 1])
def test_flatten_inference_data_to_dict(
    inference_data, var_names, groups, dimensions, group_info, var_name_format, index_origin
):
    """Test flattening (stacking) inference data (subgroups) for dictionary."""
    res_dict = flatten_inference_data_to_dict(
        data=inference_data,
        var_names=var_names,
        groups=groups,
        dimensions=dimensions,
        group_info=group_info,
        var_name_format=var_name_format,
        index_origin=index_origin,
    )
    assert res_dict
    assert "draw" in res_dict
    assert any("mu" in item for item in res_dict)
    if group_info:
        if groups != "prior_groups":
            assert any("posterior" in item for item in res_dict)
            if var_names is None:
                assert any("sample_stats" in item for item in res_dict)
        else:
            assert any("prior" in item for item in res_dict)
    else:
        if groups != "prior_groups":
            assert not any("posterior" in item for item in res_dict)
            if var_names is None:
                assert not any("sample_stats" in item for item in res_dict)
        else:
            assert not any("prior" in item for item in res_dict)
