"""Tests for stats_utils."""
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from scipy.special import logsumexp

from ..stats.stats_utils import (
    logsumexp as _logsumexp,
    make_ufunc,
    wrap_xarray_ufunc,
    check_nan,
    check_valid_size,
)


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
    np.random.seed(17)
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
    np.random.seed(17)
    ary = np.random.randn(100, 101).astype(ary_dtype)  # pylint: disable=no-member
    assert _logsumexp(ary=ary, axis=axis, b_inv=b_inv, keepdims=keepdims, copy=True) is not None
    ary = ary.copy()
    assert _logsumexp(ary=ary, axis=axis, b_inv=b_inv, keepdims=keepdims, copy=False) is not None
    out = np.empty(5)
    assert _logsumexp(ary=np.random.randn(10, 5), axis=0, out=out) is not None

    if b_inv != 0:
        # Scipy implementation when b_inv != 0
        if b_inv is not None:
            b_scipy = 1 / b_inv
        else:
            b_scipy = None
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
    else:
        if n_output == 1:
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
    assert not check_nan(np.ones(10), how=how)
    assert check_nan(np.full(10, np.nan), how=how)


def test_valid_size_bad():
    with pytest.raises(TypeError):
        check_valid_size(np.ones(10, 10), "testing", min_n_draw=100)
