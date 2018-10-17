# pylint: disable=redefined-outer-name

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_less
import pytest
from scipy.stats import linregress

from ..data import load_arviz_data
from ..stats import bfmi, compare, hpd, r2_score, waic, psislw, summary


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


def test_r2_score():
    x = np.linspace(0, 1, 100)
    y = np.random.normal(x, 1)
    res = linregress(x, y)
    assert_almost_equal(res.rvalue ** 2, r2_score(y, res.intercept + res.slope * x).r2, 2)


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_same(centered_eight, method):
    data_dict = {"first": centered_eight, "second": centered_eight}

    weight = compare(data_dict, method=method)["weight"]
    assert_almost_equal(weight[0], weight[1])
    assert_almost_equal(np.sum(weight), 1.0)


@pytest.mark.parametrize("ic", ["waic", "loo"])
@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_different(centered_eight, non_centered_eight, ic, method):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    weight = compare(model_dict, ic=ic, method=method)["weight"]
    assert weight["non_centered"] > weight["centered"]
    assert_almost_equal(np.sum(weight), 1.0)


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


def test_summary_bad_fmt(centered_eight):
    with pytest.raises(TypeError):
        summary(centered_eight, fmt="bad_fmt")


def test_waic(centered_eight):
    """Test widely available information criterion calculation"""
    assert waic(centered_eight) is not None


def test_psis():
    linewidth = np.random.randn(20000, 10)
    _, khats = psislw(linewidth)
    assert_array_less(khats, 0.5)
