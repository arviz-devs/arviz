from pandas import DataFrame
import numpy as np
import pymc3 as pm
from pytest import raises
from ..plots import densityplot, traceplot, energyplot, posteriorplot, autocorrplot, forestplot, parallelplot, pairplot


def eight_schools():
    J = 8
    y = np.asarray([28.,  8., -3.,  7., -1.,  1., 18., 12.])
    sigma = np.asarray([15., 10., 16., 11.,  9., 11., 10., 18.])
    with pm.Model() as centered_eight:
        mu = pm.Normal('mu', mu=0, sd=5)
        tau = pm.HalfCauchy('tau', beta=5)
        theta = pm.Normal('theta', mu=mu, sd=tau, shape=J)
        obs = pm.Normal('obs', mu=theta, sd=sigma, observed=y)
        short_trace = pm.sample(600, chains=2)
    return short_trace


def test_plots():
    trace0 = DataFrame({'a': np.random.rand(100)})
    trace1 = eight_schools()

    assert densityplot(trace0).shape == (1,)
    assert densityplot(trace1).shape == (10,)

    assert traceplot(trace0).shape == (1, 2)
    assert traceplot(trace1).shape == (10, 2)

    # posteriorplot(trace0).shape == (1,)
    assert posteriorplot(trace1).shape == (10,)

    assert autocorrplot(trace0).shape == (1, 1)
    assert autocorrplot(trace1).shape == (10, 2)

    assert forestplot(trace0).get_geometry() == (1, 1)
    assert forestplot(trace1).get_geometry() == (1, 2)

    with raises(AttributeError):
        energyplot(trace0)
    assert energyplot(trace1)

    with raises(ValueError):
        parallelplot(trace0)
    assert parallelplot(trace1)


def test_pairplot():
    with pm.Model() as model:
        a = pm.Normal('a', shape=2)
        c = pm.HalfNormal('c', shape=2)
        b = pm.Normal('b', a, c, shape=2)
        d = pm.Normal('d', 100, 1)
        trace = pm.sample(1000)

    pairplot(trace, varnames=['a__0', 'a__1'], divergences=True,
             marker='x', kwargs_divergences={'marker': '*', 'c': 'C'})
    pairplot(trace, divergences=True, marker='x', kwargs_divergences={'marker': '*', 'c': 'C'})
    pairplot(trace, hexbin=True, varnames=['a__0', 'a__1'],
             cmap='viridis', text_size=20)
