"""Tests for arviz.utils."""
# pylint: disable=redefined-outer-name, no-member
from unittest.mock import Mock

import numpy as np
import pytest
import scipy.stats as st

from ...data import dict_to_dataset, from_dict, load_arviz_data
from ...stats.density_utils import _circular_mean, _normalize_angle, _find_hdi_contours
from ...utils import (
    _stack,
    _subset_list,
    _var_names,
    expand_dims,
    flatten_inference_data_to_dict,
    one_de,
    two_de,
)
from ..helpers import TestRandomVariable


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
        (["ta"], ["beta1", "beta2", "theta"], "like"),
        (["~beta"], ["phi", "theta"], "like"),
        (["beta[0-9]+"], ["beta1", "beta2"], "regex"),
        (["^p"], ["phi"], "regex"),
        (["~^t"], ["beta1", "beta2", "phi"], "regex"),
    ],
)
def test_var_names_filter_multiple_input(var_args):
    samples = np.random.randn(10)
    data1 = dict_to_dataset({"beta1": samples, "beta2": samples, "phi": samples})
    data2 = dict_to_dataset({"beta1": samples, "beta2": samples, "theta": samples})
    data = [data1, data2]
    var_names, expected, filter_vars = var_args
    assert _var_names(var_names, data, filter_vars) == expected


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


def test_nonstring_var_names():
    """Check that non-string variables are preserved"""
    mu = TestRandomVariable("mu")
    samples = np.random.randn(10)
    data = dict_to_dataset({mu: samples})
    assert _var_names([mu], data) == [mu]


def test_var_names_filter_invalid_argument():
    """Check invalid argument raises."""
    samples = np.random.randn(10)
    data = dict_to_dataset({"alpha": samples})
    msg = r"^\'filter_vars\' can only be None, \'like\', or \'regex\', got: 'foo'$"
    with pytest.raises(ValueError, match=msg):
        assert _var_names(["alpha"], data, filter_vars="foo")


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

    from ... import utils

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
    from ... import utils

    @utils.conditional_jit
    def func():
        return True

    assert func()


def test_conditional_vect_numba_decorator():
    """Tests to see if Numba is used.

    Test can be distinguished from test_conditional_jit_decorator_no_numba
    by use of debugger or coverage tool
    """
    from ... import utils

    @utils.conditional_vect
    def func(a_a, b_b):
        return a_a + b_b

    value_one = np.random.randn(10)
    value_two = np.random.randn(10)
    assert np.allclose(func(value_one, value_two), value_one + value_two)


def test_conditional_vect_numba_decorator_keyword(monkeypatch):
    """Checks else statement and vect keyword argument"""
    from ... import utils

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
    elif groups == "prior_groups":
        assert all("prior" not in item for item in res_dict)

    else:
        assert all("posterior" not in item for item in res_dict)
        if var_names is None:
            assert all("sample_stats" not in item for item in res_dict)


@pytest.mark.parametrize("mean", [0, np.pi, 4 * np.pi, -2 * np.pi, -10 * np.pi])
def test_circular_mean_scipy(mean):
    """Test our `_circular_mean()` function gives same result than Scipy version."""
    rvs = st.vonmises.rvs(loc=mean, kappa=1, size=1000)
    mean_az = _circular_mean(rvs)
    mean_sp = st.circmean(rvs, low=-np.pi, high=np.pi)
    np.testing.assert_almost_equal(mean_az, mean_sp)


@pytest.mark.parametrize("mean", [0, np.pi, 4 * np.pi, -2 * np.pi, -10 * np.pi])
def test_normalize_angle(mean):
    """Testing _normalize_angles() return values between expected bounds"""
    rvs = st.vonmises.rvs(loc=mean, kappa=1, size=1000)
    values = _normalize_angle(rvs, zero_centered=True)
    assert ((-np.pi <= values) & (values <= np.pi)).all()

    values = _normalize_angle(rvs, zero_centered=False)
    assert ((values >= 0) & (values <= 2 * np.pi)).all()


@pytest.mark.parametrize("mean", [[0, 0], [1, 1]])
@pytest.mark.parametrize(
    "cov",
    [
        np.diag([1, 1]),
        np.diag([0.5, 0.5]),
        np.diag([0.25, 1]),
        np.array([[0.4, 0.2], [0.2, 0.8]]),
    ],
)
@pytest.mark.parametrize("contour_sigma", [np.array([1, 2, 3])])
def test_find_hdi_contours(mean, cov, contour_sigma):
    """Test `_find_hdi_contours()` against SciPy's multivariate normal distribution."""
    # Set up scipy distribution
    prob_dist = st.multivariate_normal(mean, cov)

    # Find standard deviations and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(cov)
    eigenvecs = eigenvecs.T
    stdevs = np.sqrt(eigenvals)

    # Find min and max for grid at 7-sigma contour
    extremes = np.empty((4, 2))
    for i in range(4):
        extremes[i] = mean + (-1) ** i * 7 * stdevs[i // 2] * eigenvecs[i // 2]
    x_min, y_min = np.amin(extremes, axis=0)
    x_max, y_max = np.amax(extremes, axis=0)

    # Create 256x256 grid
    x = np.linspace(x_min, x_max, 256)
    y = np.linspace(y_min, y_max, 256)
    grid = np.dstack(np.meshgrid(x, y))

    density = prob_dist.pdf(grid)

    contour_sp = np.empty(contour_sigma.shape)
    for idx, sigma in enumerate(contour_sigma):
        contour_sp[idx] = prob_dist.pdf(mean + sigma * stdevs[0] * eigenvecs[0])

    hdi_probs = 1 - np.exp(-0.5 * contour_sigma**2)
    contour_az = _find_hdi_contours(density, hdi_probs)

    np.testing.assert_allclose(contour_sp, contour_az, rtol=1e-2, atol=1e-4)
