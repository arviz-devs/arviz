import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_less
import pytest
from scipy.stats import linregress

from arviz import load_arviz_data
from ..stats import bfmi, compare, hpd, r2_score, waic, psislw, summary



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


class TestCompare():
    @classmethod
    def setup_class(cls):
        cls.centered = load_arviz_data('centered_eight')
        cls.non_centered = load_arviz_data('non_centered_eight')

    @pytest.mark.parametrize('method', ['stacking', 'BB-pseudo-BMA', 'pseudo-BMA'])
    def test_compare_same(self, method):
        data_dict = {
            'first': self.centered,
            'second': self.centered,
        }

        weight = compare(data_dict, method=method)['weight']
        assert_almost_equal(weight[0], weight[1])
        assert_almost_equal(np.sum(weight), 1.)

    @pytest.mark.parametrize('ic', ['waic', 'loo'])
    @pytest.mark.parametrize('method', ['stacking', 'BB-pseudo-BMA', 'pseudo-BMA'])
    def test_compare_different(self, ic, method):
        model_dict = {
            'centered': self.centered,
            'non_centered': self.non_centered
        }
        weight = compare(model_dict, ic=ic, method=method)['weight']
        assert weight['non_centered'] > weight['centered']
        assert_almost_equal(np.sum(weight), 1.)


@pytest.mark.parametrize('include_circ', [True, False])
def test_summary(include_circ):
    centered = load_arviz_data('centered_eight')
    summary(centered, include_circ=include_circ)


def test_waic():
    """Test widely available information criterion calculation"""
    centered = load_arviz_data('centered_eight')
    waic(centered)

def test_psis():
    linewidth = np.random.randn(20000, 10)
    _, khats = psislw(linewidth)
    assert_array_less(khats, .5)
