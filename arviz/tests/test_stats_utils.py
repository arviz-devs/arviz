"""Tests for stats_utils."""
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from scipy.special import logsumexp

from ..stats.stats_utils import logsumexp as _logsumexp, wrap_xarray_ufunc


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
        res = wrap_xarray_ufunc(
            np.quantile,
            ary,
            ufunc_kwargs={"n_output": n_output},
            func_kwargs={"quantile": quantile},
        )
    if n_output == 1:
        assert not isinstance(res, tuple)
    else:
        assert isinstance(res, tuple)
        assert len(res) == n_output
