# pylint: disable=redefined-outer-name
from copy import deepcopy
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_less
import pytest
from scipy.special import logsumexp
from scipy.stats import linregress


from ..data import load_arviz_data, from_dict
from ..stats import bfmi, compare, hpd, loo, r2_score, waic, psislw, summary
from ..stats.stats import _gpinv, _mc_error, _logsumexp


@pytest.fixture(scope="session")
def centered_eight():
    centered_eight = load_arviz_data("centered_eight")
    return centered_eight


@pytest.fixture(scope="session")
def non_centered_eight():
    non_centered_eight = load_arviz_data("non_centered_eight")
    return non_centered_eight


def test_bfmi():
    energy = np.array([1, 2, 3, 4])
    assert_almost_equal(bfmi(energy), 0.8)


def test_hpd():
    normal_sample = np.random.randn(5000000)
    interval = hpd(normal_sample)
    assert_array_almost_equal(interval, [-1.88, 1.88], 2)


def test_hpd_bad_ci():
    normal_sample = np.random.randn(10)
    with pytest.raises(ValueError):
        hpd(normal_sample, credible_interval=2)


def test_r2_score():
    x = np.linspace(0, 1, 100)
    y = np.random.normal(x, 1)
    res = linregress(x, y)
    assert_almost_equal(res.rvalue ** 2, r2_score(y, res.intercept + res.slope * x).r2, 2)


def test_r2_score_multivariate():
    x = np.linspace(0, 1, 100)
    y = np.random.normal(x, 1)
    res = linregress(x, y)
    y_multivariate = np.c_[y, y]
    y_multivariate_pred = np.c_[res.intercept + res.slope * x, res.intercept + res.slope * x]
    assert not np.isnan(r2_score(y_multivariate, y_multivariate_pred).r2)


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_same(centered_eight, method):
    data_dict = {"first": centered_eight, "second": centered_eight}

    weight = compare(data_dict, method=method)["weight"]
    assert_almost_equal(weight[0], weight[1])
    assert_almost_equal(np.sum(weight), 1.0)


def test_compare_unknown_ic_and_method(centered_eight, non_centered_eight):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    with pytest.raises(NotImplementedError):
        compare(model_dict, ic="Unknown", method="stacking")
    with pytest.raises(ValueError):
        compare(model_dict, ic="loo", method="Unknown")


@pytest.mark.parametrize("ic", ["waic", "loo"])
@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_different(centered_eight, non_centered_eight, ic, method):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    weight = compare(model_dict, ic=ic, method=method)["weight"]
    assert weight["non_centered"] > weight["centered"]
    assert_almost_equal(np.sum(weight), 1.0)


def test_compare_different_size(centered_eight, non_centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior = centered_eight.posterior.drop("Choate", "school")
    centered_eight.sample_stats = centered_eight.sample_stats.drop("Choate", "school")
    centered_eight.posterior_predictive = centered_eight.posterior_predictive.drop(
        "Choate", "school"
    )
    centered_eight.prior = centered_eight.prior.drop("Choate", "school")
    centered_eight.observed_data = centered_eight.observed_data.drop("Choate", "school")
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    with pytest.raises(ValueError):
        compare(model_dict, ic="waic", method="stacking")


@pytest.mark.parametrize("var_names_expected", ((None, 10), ("mu", 1), (["mu", "tau"], 2)))
def test_summary_var_names(var_names_expected):
    var_names, expected = var_names_expected
    centered = load_arviz_data("centered_eight")
    summary_df = summary(centered, var_names=var_names)
    assert len(summary_df.index) == expected


@pytest.mark.parametrize("include_circ", [True, False])
def test_summary_include_circ(centered_eight, include_circ):
    assert summary(centered_eight, include_circ=include_circ) is not None


@pytest.mark.parametrize("fmt", ["wide", "long", "xarray"])
def test_summary_fmt(centered_eight, fmt):
    assert summary(centered_eight, fmt=fmt) is not None


@pytest.mark.parametrize("order", ["C", "F"])
def test_summary_unpack_order(order):
    data = from_dict({"a": np.random.randn(4, 100, 4, 5, 3)})
    az_summary = summary(data, order=order, fmt="wide")
    assert az_summary is not None
    if order != "F":
        first_index = 4
        second_index = 5
        third_index = 3
    else:
        first_index = 3
        second_index = 5
        third_index = 4
    column_order = []
    for idx1 in range(first_index):
        for idx2 in range(second_index):
            for idx3 in range(third_index):
                if order != "F":
                    column_order.append("a[{},{},{}]".format(idx1, idx2, idx3))
                else:
                    column_order.append("a[{},{},{}]".format(idx3, idx2, idx1))
    for col1, col2 in zip(list(az_summary.index), column_order):
        assert col1 == col2


def test_summary_stat_func(centered_eight):
    assert summary(centered_eight, stat_funcs=[np.var]) is not None


def test_summary_nan(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior.theta[:, :, 0] = np.nan
    summary_xarray = summary(centered_eight)
    assert summary_xarray is not None
    assert summary_xarray.loc["theta[0]"].isnull().all()
    assert (
        summary_xarray.loc[[ix for ix in summary_xarray.index if ix != "theta[0]"]]
        .notnull()
        .all()
        .all()
    )


@pytest.mark.parametrize("fmt", [1, "bad_fmt"])
def test_summary_bad_fmt(centered_eight, fmt):
    with pytest.raises(TypeError):
        summary(centered_eight, fmt=fmt)


@pytest.mark.parametrize("order", [1, "bad_order"])
def test_summary_bad_unpack_order(centered_eight, order):
    with pytest.raises(TypeError):
        summary(centered_eight, order=order)


def test_waic(centered_eight):
    """Test widely available information criterion calculation"""
    assert waic(centered_eight) is not None
    assert waic(centered_eight, pointwise=True) is not None


def test_waic_bad(centered_eight):
    """Test widely available information criterion calculation"""
    centered_eight = deepcopy(centered_eight)
    del centered_eight.sample_stats["log_likelihood"]
    with pytest.raises(TypeError):
        waic(centered_eight)

    del centered_eight.sample_stats
    with pytest.raises(TypeError):
        waic(centered_eight)


def test_waic_warning(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.sample_stats["log_likelihood"][:, :250, 1] = 10
    with pytest.warns(UserWarning):
        assert waic(centered_eight, pointwise=True) is not None
    # this should throw a warning, but due to numerical issues it fails
    centered_eight.sample_stats["log_likelihood"][:, :, :] = 0
    with pytest.warns(UserWarning):
        assert waic(centered_eight, pointwise=True) is not None


def test_loo(centered_eight):
    assert loo(centered_eight) is not None


def test_loo_one_chain(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior = centered_eight.posterior.drop([1, 2, 3], "chain")
    centered_eight.sample_stats = centered_eight.sample_stats.drop([1, 2, 3], "chain")
    assert loo(centered_eight) is not None


def test_loo_pointwise(centered_eight):
    assert loo(centered_eight, pointwise=True) is not None


def test_loo_bad(centered_eight):
    with pytest.raises(TypeError):
        loo(np.random.randn(2, 10))

    centered_eight = deepcopy(centered_eight)
    del centered_eight.sample_stats["log_likelihood"]
    with pytest.raises(TypeError):
        loo(centered_eight)


def test_loo_warning(centered_eight):
    centered_eight = deepcopy(centered_eight)
    # make one of the khats infinity
    centered_eight.sample_stats["log_likelihood"][:, :, 1] = 10
    with pytest.warns(UserWarning):
        assert loo(centered_eight, pointwise=True) is not None
    # make all of the khats infinity
    centered_eight.sample_stats["log_likelihood"][:, :, :] = 0
    with pytest.warns(UserWarning):
        assert loo(centered_eight, pointwise=True) is not None


def test_psislw():
    linewidth = np.random.randn(20000, 10)
    _, khats = psislw(linewidth)
    assert_array_less(khats, 0.5)


@pytest.mark.parametrize("size", [100, 101])
@pytest.mark.parametrize("batches", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("circular", [False, True])
def test_mc_error(size, batches, ndim, circular):
    x = np.random.randn(size, ndim).squeeze()
    assert _mc_error(x, batches=batches, circular=circular) is not None


@pytest.mark.parametrize("probs", [True, False])
@pytest.mark.parametrize("kappa", [-1, -0.5, 1e-30, 0.5, 1])
@pytest.mark.parametrize("sigma", [0, 2])
def test_gpinv(probs, kappa, sigma):
    if probs:
        probs = np.array([0.1, 0.1, 0.1, 0.2, 0.3])
    else:
        probs = np.array([-0.1, 0.1, 0.1, 0.2, 0.3])
    assert len(_gpinv(probs, kappa, sigma)) == len(probs)


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
    ary = np.random.randn(100, 101).astype(ary_dtype)
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
    ary = np.random.randn(100, 101).astype(ary_dtype)
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
