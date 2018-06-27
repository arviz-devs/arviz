import copy

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_less
import pandas as pd
import pymc3 as pm
from scipy import stats

from ..stats import bfmi, compare, hpd, r2_score, summary, waic, psislw



def test_bfmi():
    trace = pd.DataFrame([1, 2, 3, 4], columns=['energy'])
    assert_almost_equal(bfmi(trace), 0.8)


def test_hpd():
    normal_sample = np.random.randn(5000000)
    interval = hpd(normal_sample)
    assert_array_almost_equal(interval, [-1.96, 1.96], 2)


def test_r2_score():
    x = np.linspace(0, 1, 100)
    y = np.random.normal(x, 1)
    res = stats.linregress(x, y)
    assert_almost_equal(res.rvalue ** 2, r2_score(y, res.intercept + res.slope * x).r2, 2)


class TestCompare(object):
    @classmethod
    def setup_class(cls):
        np.random.seed(42)
        x_obs = np.random.normal(0, 1, size=100)

        with pm.Model() as cls.model0:
            mu = pm.Normal('mu', 0, 1)
            pm.Normal('x', mu=mu, sd=1, observed=x_obs)
            cls.trace0 = pm.sample(1000)

        with pm.Model() as cls.model1:
            mu = pm.Normal('mu', 0, 1)
            pm.Normal('x', mu=mu, sd=0.8, observed=x_obs)
            cls.trace1 = pm.sample(1000)

        with pm.Model() as cls.model2:
            mu = pm.Normal('mu', 0, 1)
            pm.StudentT('x', nu=1, mu=mu, lam=1, observed=x_obs)
            cls.trace2 = pm.sample(1000)

    def test_compare_same(self):
        traces = [self.trace0, copy.copy(self.trace0)]
        models = [self.model0, copy.copy(self.model0)]

        model_dict = dict(zip(models, traces))

        for method in ('stacking', 'BB-pseudo-BMA', 'pseudo-BMA'):
            weight = compare(model_dict, method=method)['weight']
            assert_almost_equal(weight[0], weight[1])
            assert_almost_equal(np.sum(weight), 1.)

    def test_compare_different(self):
        model_dict = {
            self.model0: self.trace0,
            self.model1: self.trace1,
            self.model2: self.trace2,
        }
        for method in ('stacking', 'BB-pseudo-BMA', 'pseudo-BMA'):
            weight = compare(model_dict, method=method)['weight']
            assert weight[0] > weight[1] > weight[2]
            assert_almost_equal(np.sum(weight), 1.)


def test_summary():
    alpha = np.repeat((1, 5, 10), 100)
    beta = np.repeat((1, 5, 1), 100)
    data = np.random.beta(alpha, beta).reshape(-1, 50)
    trace = pd.DataFrame(data.T, columns=['a', 'a', 'b', 'b', 'c', 'c'])

    df_s = summary(trace)
    assert df_s.shape == (3, 7)
    assert np.all(df_s.index == ['a', 'b', 'c'])


def test_waic():
    """Test widely available information criterion calculation"""
    x_obs = np.arange(6)

    with pm.Model() as model:
        prob = pm.Beta('p', 1., 1., transform=None)
        pm.Binomial('x', 5, prob, observed=x_obs)

        step = pm.Metropolis()
        trace = pm.sample(100, step)

    calculated_waic = waic(trace, model)
    log_py = stats.binom.logpmf(np.atleast_2d(x_obs).T, 5, trace['p']).T

    lppd_i = np.log(np.mean(np.exp(log_py), axis=0))
    vars_lpd = np.var(log_py, axis=0)
    waic_i = - 2 * (lppd_i - vars_lpd)

    actual_waic_se = np.sqrt(len(waic_i) * np.var(waic_i))
    actual_waic = np.sum(waic_i)

    assert_almost_equal(np.asarray(calculated_waic.waic),
                        actual_waic, decimal=2)
    assert_almost_equal(np.asarray(calculated_waic.waic_se),
                        actual_waic_se, decimal=2)

def test_psis():
    linewidth = np.random.randn(20000, 10)
    _, khats = psislw(linewidth)
    assert_array_less(khats, .5)
