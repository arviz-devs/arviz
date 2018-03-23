import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats
import copy
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from ..stats import bfmi, compare, hpd, r2_score, summary, waic


def fake_trace(n_samples):
    a = np.repeat((1, 5, 10), n_samples)
    b = np.repeat((1, 5, 1), n_samples)
    data = np.random.beta(a, b).reshape(-1, n_samples//2)
    trace = pd.DataFrame(data.T, columns=['a', 'a', 'b', 'b', 'c', 'c'])
    return trace


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
    assert_almost_equal(res.rvalue ** 2, r2_score(y, res.intercept + res.slope * x).r2_median, 2)


def test_compare():
    np.random.seed(42)
    x_obs = np.random.normal(0, 1, size=100)

    with pm.Model() as model0:
        mu = pm.Normal('mu', 0, 1)
        x = pm.Normal('x', mu=mu, sd=1, observed=x_obs)
        trace0 = pm.sample(1000)

    with pm.Model() as model1:
        mu = pm.Normal('mu', 0, 1)
        x = pm.Normal('x', mu=mu, sd=0.8, observed=x_obs)
        trace1 = pm.sample(1000)

    with pm.Model() as model2:
        mu = pm.Normal('mu', 0, 1)
        x = pm.StudentT('x', nu=1, mu=mu, lam=1, observed=x_obs)
        trace2 = pm.sample(1000)

    traces = [trace0, copy.copy(trace0)]
    models = [model0, copy.copy(model0)]

    model_dict = dict(zip(models, traces))

    w_st = pm.compare(model_dict, method='stacking')['weight']
    w_bb_bma = pm.compare(model_dict, method='BB-pseudo-BMA')['weight']
    w_bma = pm.compare(model_dict, method='pseudo-BMA')['weight']

    assert_almost_equal(w_st[0], w_st[1])
    assert_almost_equal(w_bb_bma[0], w_bb_bma[1])
    assert_almost_equal(w_bma[0], w_bma[1])

    assert_almost_equal(np.sum(w_st), 1.)
    assert_almost_equal(np.sum(w_bb_bma), 1.)
    assert_almost_equal(np.sum(w_bma), 1.)

    traces = [trace0, trace1, trace2]
    models = [model0, model1, model2]

    model_dict = dict(zip(models, traces))

    w_st = pm.compare(model_dict, method='stacking')['weight']
    w_bb_bma = pm.compare(model_dict, method='BB-pseudo-BMA')['weight']
    w_bma = pm.compare(model_dict, method='pseudo-BMA')['weight']

    assert(w_st[0] > w_st[1] > w_st[2])
    assert(w_bb_bma[0] > w_bb_bma[1] > w_bb_bma[2])
    assert(w_bma[0] > w_bma[1] > w_bma[2])

    assert_almost_equal(np.sum(w_st), 1.)
    assert_almost_equal(np.sum(w_st), 1.)
    assert_almost_equal(np.sum(w_st), 1.)


def test_summary():
    trace = fake_trace(100)
    df_s = summary(trace)
    assert df_s.shape == (3, 7)
    assert np.all(df_s.index == ['a', 'b', 'c'])


def test_waic():
    """Test widely available information criterion calculation"""
    x_obs = np.arange(6)

    with pm.Model():
        p = pm.Beta('p', 1., 1., transform=None)
        pm.Binomial('x', 5, p, observed=x_obs)

        step = pm.Metropolis()
        trace = pm.sample(100, step)
        calculated_waic = pm.waic(trace)

    log_py = stats.binom.logpmf(np.atleast_2d(x_obs).T, 5, trace['p']).T

    lppd_i = np.log(np.mean(np.exp(log_py), axis=0))
    vars_lpd = np.var(log_py, axis=0)
    waic_i = - 2 * (lppd_i - vars_lpd)

    actual_waic_se = np.sqrt(len(waic_i) * np.var(waic_i))
    actual_waic = np.sum(waic_i)

    assert_almost_equal(calculated_waic.WAIC, actual_waic, decimal=2)
    assert_almost_equal(calculated_waic.WAIC_se, actual_waic_se, decimal=2)
