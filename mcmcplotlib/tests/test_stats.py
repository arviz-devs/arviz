import numpy as np
from scipy import stats
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from ..stats import hpd, r2_score


def test_hpd():
    normal_sample = np.random.randn(5000000)
    interval = hpd(normal_sample)
    assert_array_almost_equal(interval, [-1.96, 1.96], 2)


def test_r2_score():
    x = np.linspace(0, 1, 100)
    y = np.random.normal(x, 1)
    res = stats.linregress(x, y)
    assert_almost_equal(res.rvalue ** 2, r2_score(y, res.intercept + res.slope * x).r2_median, 2)


def test_r2_score():
    x = np.linspace(0, 1, 100)
    y = np.random.normal(x, 1)
    res = stats.linregress(x, y)
    assert_almost_equal(res.rvalue ** 2, r2_score(y, res.intercept + res.slope * x).r2_median, 2)
