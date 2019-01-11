# pylint: disable=redefined-outer-name
from copy import deepcopy
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_less
import pytest
from scipy.stats import linregress

from ..data import load_arviz_data
from ..stats import bfmi, compare, hpd, loo, r2_score, waic, psislw, summary
from ..stats.stats import _gpinv, _mc_error


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


# Issue with xarray.apply_ufunc
@pytest.mark.xfail(
    reason="Issue #507. "
    "_make_ufunc is broken."
    "See https://github.com/arviz-devs/arviz/issues/507"
)
def test_summary_stat_func(centered_eight):
    assert summary(centered_eight, stat_funcs=[np.var]) is not None


def test_summary_bad_fmt(centered_eight):
    with pytest.raises(TypeError):
        summary(centered_eight, fmt="bad_fmt")


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


@pytest.mark.xfail(
    reason="Issue #509. "
    "Numerical accuracy (logsumexp) prevents function to throw a warning."
    "See https://github.com/arviz-devs/arviz/issues/509"
)
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
