from pandas import DataFrame
import numpy as np
import pymc3 as pm
from pytest import raises
from ..plots import (densityplot, traceplot, energyplot, posteriorplot, autocorrplot, forestplot,
                     parallelplot, pairplot, jointplot)


J = 8
y = np.asarray([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.asarray([15., 10., 16., 11.,  9., 11., 10., 18.])
with pm.Model() as centered_eight:
    mu = pm.Normal('mu', mu=0, sd=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta = pm.Normal('theta', mu=mu, sd=tau, shape=J)
    obs = pm.Normal('obs', mu=theta, sd=sigma, observed=y)
    short_trace = pm.sample(600, chains=2)


def test_plots():
    trace0 = DataFrame({'a': np.random.poisson(2.3, 100)})

    assert densityplot(trace0).shape == (1,)
    assert densityplot(short_trace).shape == (10,)

    assert traceplot(trace0).shape == (1, 2)
    assert traceplot(short_trace).shape == (10, 2)

    # posteriorplot(trace0).shape == (1,)
    assert posteriorplot(short_trace).shape == (10,)

    assert autocorrplot(trace0).shape == (1, 1)
    assert autocorrplot(short_trace).shape == (10, 2)

    assert forestplot(trace0).get_geometry() == (1, 1)
    assert forestplot(short_trace).get_geometry() == (1, 3)
    assert forestplot(short_trace, rhat=False).get_geometry() == (1, 2)
    assert forestplot(short_trace, neff=False).get_geometry() == (1, 2)

    with raises(AttributeError):
        energyplot(trace0)
    assert energyplot(short_trace)

    with raises(ValueError):
        parallelplot(trace0)
    assert parallelplot(short_trace)

    jointplot(short_trace, varnames=['mu', 'tau'])


def test_pairplot():
    pairplot(short_trace, varnames=['theta__0', 'theta__1'], divergences=True,
             marker='x', kwargs_divergences={'marker': '*', 'c': 'C'})
    pairplot(short_trace, divergences=True, marker='x',
             kwargs_divergences={'marker': '*', 'c': 'C'})
    pairplot(short_trace, kind='hexbin', varnames=['theta__0', 'theta__1'],
             cmap='viridis', textsize=20)
