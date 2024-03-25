"""Tests for stats_utils."""

#  pylint: disable=no-member
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.special import logsumexp
from scipy.stats import circstd

from ...data import from_dict, load_arviz_data
from ...stats.density_utils import histogram
from ...stats.stats_utils import (
    ELPDData,
    _angle,
    _circfunc,
    _circular_standard_deviation,
    _sqrt,
    get_log_likelihood,
)
from ...stats.stats_utils import logsumexp as _logsumexp
from ...stats.stats_utils import make_ufunc, not_valid, stats_variance_2d, wrap_xarray_ufunc


@pytest.mark.parametrize("ary_dtype", [np.float64, np.float32, np.int32, np.int64])
@pytest.mark.parametrize("axis", [None, 0, 1, (-2, -1)])
@pytest.mark.parametrize("b", [None, 0, 1 / 100, 1 / 101])
@pytest.mark.parametrize("keepdims", [True, False])
def test_logsumexp_b(ary_dtype, axis, b, keepdims):
    """Test ArviZ implementation of logsumexp.

    Test also compares against Scipy implementation.
    Case where b=None, they are equal. (N=len(ary))
    Second case where b=x, and x is 1/(number of elements), they are almost equal.

    Test tests against b parameter.
    """
    ary = np.random.randn(100, 101).astype(ary_dtype)  # pylint: disable=no-member
    assert _logsumexp(ary=ary, axis=axis, b=b, keepdims=keepdims, copy=True) is not None
    ary = ary.copy()
    assert _logsumexp(ary=ary, axis=axis, b=b, keepdims=keepdims, copy=False) is not None
    out = np.empty(5)
    assert _logsumexp(ary=np.random.randn(10, 5), axis=0, out=out) is not None

    # Scipy implementation
    scipy_results = logsumexp(ary, b=b, axis=axis, keepdims=keepdims)
    arviz_results = _logsumexp(ary, b=b, axis=axis, keepdims=keepdims)

    assert_array_almost_equal(scipy_results, arviz_results)


@pytest.mark.parametrize("ary_dtype", [np.float64, np.float32, np.int32, np.int64])
@pytest.mark.parametrize("axis", [None, 0, 1, (-2, -1)])
@pytest.mark.parametrize("b_inv", [None, 0, 100, 101])
@pytest.mark.parametrize("keepdims", [True, False])
def test_logsumexp_b_inv(ary_dtype, axis, b_inv, keepdims):
    """Test ArviZ implementation of logsumexp.

    Test also compares against Scipy implementation.
    Case where b=None, they are equal. (N=len(ary))
    Second case where b=x, and x is 1/(number of elements), they are almost equal.

    Test tests against b_inv parameter.
    """
    ary = np.random.randn(100, 101).astype(ary_dtype)  # pylint: disable=no-member
    assert _logsumexp(ary=ary, axis=axis, b_inv=b_inv, keepdims=keepdims, copy=True) is not None
    ary = ary.copy()
    assert _logsumexp(ary=ary, axis=axis, b_inv=b_inv, keepdims=keepdims, copy=False) is not None
    out = np.empty(5)
    assert _logsumexp(ary=np.random.randn(10, 5), axis=0, out=out) is not None

    if b_inv != 0:
        # Scipy implementation when b_inv != 0
        b_scipy = 1 / b_inv if b_inv is not None else None
        scipy_results = logsumexp(ary, b=b_scipy, axis=axis, keepdims=keepdims)
        arviz_results = _logsumexp(ary, b_inv=b_inv, axis=axis, keepdims=keepdims)

        assert_array_almost_equal(scipy_results, arviz_results)


@pytest.mark.parametrize("quantile", ((0.5,), (0.5, 0.1)))
@pytest.mark.parametrize("arg", (True, False))
def test_wrap_ufunc_output(quantile, arg):
    ary = np.random.randn(4, 100)
    n_output = len(quantile)
    if arg:
        res = wrap_xarray_ufunc(
            np.quantile, ary, ufunc_kwargs={"n_output": n_output}, func_args=(quantile,)
        )
    elif n_output == 1:
        res = wrap_xarray_ufunc(np.quantile, ary, func_kwargs={"q": quantile})
    else:
        res = wrap_xarray_ufunc(
            np.quantile, ary, ufunc_kwargs={"n_output": n_output}, func_kwargs={"q": quantile}
        )
    if n_output == 1:
        assert not isinstance(res, tuple)
    else:
        assert isinstance(res, tuple)
        assert len(res) == n_output


@pytest.mark.parametrize("out_shape", ((1, 2), (1, 2, 3), (2, 3, 4, 5)))
@pytest.mark.parametrize("input_dim", ((4, 100), (4, 100, 3), (4, 100, 4, 5)))
def test_wrap_ufunc_out_shape(out_shape, input_dim):
    func = lambda x: np.random.rand(*out_shape)
    ary = np.ones(input_dim)
    res = wrap_xarray_ufunc(
        func, ary, func_kwargs={"out_shape": out_shape}, ufunc_kwargs={"n_dims": 1}
    )
    assert res.shape == (*ary.shape[:-1], *out_shape)


def test_wrap_ufunc_out_shape_multi_input():
    out_shape = (2, 4)
    func = lambda x, y: np.random.rand(*out_shape)
    ary1 = np.ones((4, 100))
    ary2 = np.ones((4, 5))
    res = wrap_xarray_ufunc(
        func, ary1, ary2, func_kwargs={"out_shape": out_shape}, ufunc_kwargs={"n_dims": 1}
    )
    assert res.shape == (*ary1.shape[:-1], *out_shape)


def test_wrap_ufunc_out_shape_multi_output_same():
    func = lambda x: (np.random.rand(1, 2), np.random.rand(1, 2))
    ary = np.ones((4, 100))
    res1, res2 = wrap_xarray_ufunc(
        func,
        ary,
        func_kwargs={"out_shape": ((1, 2), (1, 2))},
        ufunc_kwargs={"n_dims": 1, "n_output": 2},
    )
    assert res1.shape == (*ary.shape[:-1], 1, 2)
    assert res2.shape == (*ary.shape[:-1], 1, 2)


def test_wrap_ufunc_out_shape_multi_output_diff():
    func = lambda x: (np.random.rand(5, 3), np.random.rand(10, 4))
    ary = np.ones((4, 100))
    res1, res2 = wrap_xarray_ufunc(
        func,
        ary,
        func_kwargs={"out_shape": ((5, 3), (10, 4))},
        ufunc_kwargs={"n_dims": 1, "n_output": 2},
    )
    assert res1.shape == (*ary.shape[:-1], 5, 3)
    assert res2.shape == (*ary.shape[:-1], 10, 4)


@pytest.mark.parametrize("n_output", (1, 2, 3))
def test_make_ufunc(n_output):
    if n_output == 3:
        func = lambda x: (np.mean(x), np.mean(x), np.mean(x))
    elif n_output == 2:
        func = lambda x: (np.mean(x), np.mean(x))
    else:
        func = np.mean
    ufunc = make_ufunc(func, n_dims=1, n_output=n_output)
    ary = np.ones((4, 100))
    res = ufunc(ary)
    if n_output > 1:
        assert all(len(res_i) == 4 for res_i in res)
        assert all((res_i == 1).all() for res_i in res)
    else:
        assert len(res) == 4
        assert (res == 1).all()


@pytest.mark.parametrize("n_output", (1, 2, 3))
def test_make_ufunc_out(n_output):
    if n_output == 3:
        func = lambda x: (np.mean(x), np.mean(x), np.mean(x))
        res = (np.empty((4,)), np.empty((4,)), np.empty((4,)))
    elif n_output == 2:
        func = lambda x: (np.mean(x), np.mean(x))
        res = (np.empty((4,)), np.empty((4,)))
    else:
        func = np.mean
        res = np.empty((4,))
    ufunc = make_ufunc(func, n_dims=1, n_output=n_output)
    ary = np.ones((4, 100))
    ufunc(ary, out=res)
    if n_output > 1:
        assert all(len(res_i) == 4 for res_i in res)
        assert all((res_i == 1).all() for res_i in res)
    else:
        assert len(res) == 4
        assert (res == 1).all()


def test_make_ufunc_bad_ndim():
    with pytest.raises(TypeError):
        make_ufunc(np.mean, n_dims=0)


@pytest.mark.parametrize("n_output", (1, 2, 3))
def test_make_ufunc_out_bad(n_output):
    if n_output == 3:
        func = lambda x: (np.mean(x), np.mean(x), np.mean(x))
        res = (np.empty((100,)), np.empty((100,)))
    elif n_output == 2:
        func = lambda x: (np.mean(x), np.mean(x))
        res = np.empty((100,))
    else:
        func = np.mean
        res = np.empty((100,))
    ufunc = make_ufunc(func, n_dims=1, n_output=n_output)
    ary = np.ones((4, 100))
    with pytest.raises(TypeError):
        ufunc(ary, out=res)


@pytest.mark.parametrize("how", ("all", "any"))
def test_nan(how):
    assert not not_valid(np.ones(10), check_shape=False, nan_kwargs=dict(how=how))
    if how == "any":
        assert not_valid(
            np.concatenate((np.random.randn(100), np.full(2, np.nan))),
            check_shape=False,
            nan_kwargs=dict(how=how),
        )
    else:
        assert not not_valid(
            np.concatenate((np.random.randn(100), np.full(2, np.nan))),
            check_shape=False,
            nan_kwargs=dict(how=how),
        )
        assert not_valid(np.full(10, np.nan), check_shape=False, nan_kwargs=dict(how=how))


@pytest.mark.parametrize("axis", (-1, 0, 1))
def test_nan_axis(axis):
    data = np.random.randn(4, 100)
    data[0, 0] = np.nan  #  pylint: disable=unsupported-assignment-operation
    axis_ = (len(data.shape) + axis) if axis < 0 else axis
    assert not_valid(data, check_shape=False, nan_kwargs=dict(how="any"))
    assert not_valid(data, check_shape=False, nan_kwargs=dict(how="any", axis=axis)).any()
    assert not not_valid(data, check_shape=False, nan_kwargs=dict(how="any", axis=axis)).all()
    assert not_valid(data, check_shape=False, nan_kwargs=dict(how="any", axis=axis)).shape == tuple(
        dim for ax, dim in enumerate(data.shape) if ax != axis_
    )


def test_valid_shape():
    assert not not_valid(
        np.ones((2, 200)), check_nan=False, shape_kwargs=dict(min_chains=2, min_draws=100)
    )
    assert not not_valid(
        np.ones((200, 2)), check_nan=False, shape_kwargs=dict(min_chains=100, min_draws=2)
    )
    assert not_valid(
        np.ones((10, 10)), check_nan=False, shape_kwargs=dict(min_chains=2, min_draws=100)
    )
    assert not_valid(
        np.ones((10, 10)), check_nan=False, shape_kwargs=dict(min_chains=100, min_draws=2)
    )


def test_get_log_likelihood():
    idata = from_dict(
        log_likelihood={
            "y1": np.random.normal(size=(4, 100, 6)),
            "y2": np.random.normal(size=(4, 100, 8)),
        }
    )
    lik1 = get_log_likelihood(idata, "y1")
    lik2 = get_log_likelihood(idata, "y2")
    assert lik1.shape == (4, 100, 6)
    assert lik2.shape == (4, 100, 8)


def test_get_log_likelihood_warning():
    idata = from_dict(
        sample_stats={
            "log_likelihood": np.random.normal(size=(4, 100, 6)),
        }
    )
    with pytest.warns(DeprecationWarning):
        get_log_likelihood(idata)


def test_get_log_likelihood_no_var_name():
    idata = from_dict(
        log_likelihood={
            "y1": np.random.normal(size=(4, 100, 6)),
            "y2": np.random.normal(size=(4, 100, 8)),
        }
    )
    with pytest.raises(TypeError, match="Found several"):
        get_log_likelihood(idata)


def test_get_log_likelihood_no_group():
    idata = from_dict(
        posterior={
            "a": np.random.normal(size=(4, 100)),
            "b": np.random.normal(size=(4, 100)),
        }
    )
    with pytest.raises(TypeError, match="log likelihood not found"):
        get_log_likelihood(idata)


def test_elpd_data_error():
    with pytest.raises(IndexError):
        repr(ELPDData(data=[0, 1, 2], index=["not IC", "se", "p"]))


def test_stats_variance_1d():
    """Test for stats_variance_1d."""
    data = np.random.rand(1000000)
    assert np.allclose(np.var(data), stats_variance_2d(data))
    assert np.allclose(np.var(data, ddof=1), stats_variance_2d(data, ddof=1))


def test_stats_variance_2d():
    """Test for stats_variance_2d."""
    data_1 = np.random.randn(1000, 1000)
    data_2 = np.random.randn(1000000)
    school = load_arviz_data("centered_eight").posterior["mu"].values
    n_school = load_arviz_data("non_centered_eight").posterior["mu"].values
    assert np.allclose(np.var(school, ddof=1, axis=1), stats_variance_2d(school, ddof=1, axis=1))
    assert np.allclose(np.var(school, ddof=1, axis=0), stats_variance_2d(school, ddof=1, axis=0))
    assert np.allclose(
        np.var(n_school, ddof=1, axis=1), stats_variance_2d(n_school, ddof=1, axis=1)
    )
    assert np.allclose(
        np.var(n_school, ddof=1, axis=0), stats_variance_2d(n_school, ddof=1, axis=0)
    )
    assert np.allclose(np.var(data_2), stats_variance_2d(data_2))
    assert np.allclose(np.var(data_2, ddof=1), stats_variance_2d(data_2, ddof=1))
    assert np.allclose(np.var(data_1, axis=0), stats_variance_2d(data_1, axis=0))
    assert np.allclose(np.var(data_1, axis=1), stats_variance_2d(data_1, axis=1))
    assert np.allclose(np.var(data_1, axis=0, ddof=1), stats_variance_2d(data_1, axis=0, ddof=1))
    assert np.allclose(np.var(data_1, axis=1, ddof=1), stats_variance_2d(data_1, axis=1, ddof=1))


def test_variance_bad_data():
    """Test for variance when the data range is extremely wide."""
    data = np.array([1e20, 200e-10, 1e-17, 432e9, 2500432, 23e5, 16e-7])
    assert np.allclose(stats_variance_2d(data), np.var(data))
    assert np.allclose(stats_variance_2d(data, ddof=1), np.var(data, ddof=1))
    assert not np.allclose(stats_variance_2d(data), np.var(data, ddof=1))


def test_histogram():
    school = load_arviz_data("non_centered_eight").posterior["mu"].values
    k_count_az, k_dens_az, _ = histogram(school, bins=np.asarray([-np.inf, 0.5, 0.7, 1, np.inf]))
    k_dens_np, *_ = np.histogram(school, bins=[-np.inf, 0.5, 0.7, 1, np.inf], density=True)
    k_count_np, *_ = np.histogram(school, bins=[-np.inf, 0.5, 0.7, 1, np.inf], density=False)
    assert np.allclose(k_count_az, k_count_np)
    assert np.allclose(k_dens_az, k_dens_np)


def test_sqrt():
    x = np.random.rand(100)
    y = np.random.rand(100)
    assert np.allclose(_sqrt(x, y), np.sqrt(x + y))


def test_angle():
    x = np.random.randn(100)
    high = 8
    low = 4
    res = (x - low) * 2 * np.pi / (high - low)
    assert np.allclose(_angle(x, low, high, np.pi), res)


def test_circfunc():
    school = load_arviz_data("centered_eight").posterior["mu"].values
    a_a = _circfunc(school, 8, 4, skipna=False)
    assert np.allclose(a_a, _angle(school, 4, 8, np.pi))


@pytest.mark.parametrize(
    "data", (np.random.randn(100), np.random.randn(100, 100), np.random.randn(100, 100, 100))
)
def test_circular_standard_deviation_1d(data):
    high = 8
    low = 4
    assert np.allclose(
        _circular_standard_deviation(data, high=high, low=low),
        circstd(data, high=high, low=low),
    )
