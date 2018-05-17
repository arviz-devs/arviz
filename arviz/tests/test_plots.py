from pandas import DataFrame
import numpy as np
import pymc3 as pm
from pytest import raises

from ..plots import (densityplot, traceplot, energyplot, posteriorplot, autocorrplot, forestplot,
                     parallelplot, pairplot, jointplot, ppcplot)


class TestPlots(object):
    @classmethod
    def setup_class(cls):
        num_schools = 8
        cls.y = np.asarray([28., 8., -3., 7., -1., 1., 18., 12.])
        sigma = np.asarray([15., 10., 16., 11., 9., 11., 10., 18.])
        with pm.Model():
            mu = pm.Normal('mu', mu=0, sd=5)
            tau = pm.HalfCauchy('tau', beta=5)
            theta = pm.Normal('theta', mu=mu, sd=tau, shape=num_schools)
            pm.Normal('obs', mu=theta, sd=sigma, observed=cls.y)
            cls.short_trace = pm.sample(600, chains=2)
            cls.sample_ppc = pm.sample_ppc(cls.short_trace, 100)
        cls.df_trace = DataFrame({'a': np.random.poisson(2.3, 100)})


    def test_density_plot(self):
        assert densityplot(self.df_trace).shape == (1,)
        assert densityplot(self.short_trace).shape == (10,)

    def test_traceplot(self):
        assert traceplot(self.df_trace).shape == (1, 2)
        assert traceplot(self.short_trace).shape == (10, 2)

    def test_posteriorplot(self):
        # posteriorplot(self.df_trace).shape == (1,)
        assert posteriorplot(self.short_trace).shape == (10,)

    def test_autocorrplot(self):
        assert autocorrplot(self.df_trace).shape == (1, 1)
        assert autocorrplot(self.short_trace).shape == (10, 2)

    def test_forestplot(self):
        assert forestplot(self.df_trace).get_geometry() == (1, 1)
        assert forestplot(self.short_trace).get_geometry() == (1, 3)
        assert forestplot(self.short_trace, rhat=False).get_geometry() == (1, 2)
        assert forestplot(self.short_trace, neff=False).get_geometry() == (1, 2)

    def test_energyplot(self):
        with raises(AttributeError):
            energyplot(self.df_trace)
        assert energyplot(self.short_trace)

    def test_parallelplot(self):
        with raises(ValueError):
            parallelplot(self.df_trace)
        assert parallelplot(self.short_trace)

    def test_jointplot(self):
        jointplot(self.short_trace, varnames=['mu', 'tau'])

    def test_pairplot(self):
        pairplot(self.short_trace, varnames=['theta__0', 'theta__1'], divergences=True,
                 marker='x', kwargs_divergences={'marker': '*', 'c': 'C'})
        pairplot(self.short_trace, divergences=True, marker='x',
                 kwargs_divergences={'marker': '*', 'c': 'C'})
        pairplot(self.short_trace, kind='hexbin', varnames=['theta__0', 'theta__1'],
                 cmap='viridis', textsize=20)

    def test_ppcplot(self):
        ppcplot(self.y, self.sample_ppc)
