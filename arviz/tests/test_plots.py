from pandas import DataFrame
import numpy as np
import pymc3 as pm
from pytest import raises

from .helpers import eight_schools_params, load_cached_models
from ..plots import (densityplot, traceplot, energyplot, posteriorplot, autocorrplot, forestplot,
                     parallelplot, pairplot, jointplot, ppcplot, violintraceplot)


class TestPlots(object):
    @classmethod
    def setup_class(cls):
        cls.data = eight_schools_params()
        models = load_cached_models(draws=500, chains=2)
        model, cls.short_trace = models['pymc3']
        with model:
            cls.sample_ppc = pm.sample_ppc(cls.short_trace, 100)
        cls.stan_model, cls.fit = models['pystan']
        cls.df_trace = DataFrame({'a': np.random.poisson(2.3, 100)})


    def test_density_plot(self):
        for obj in (self.short_trace, self.fit):
            assert densityplot(obj).shape == (18, 1)

    def test_traceplot(self):
        assert traceplot(self.df_trace).shape == (1, 2)
        assert traceplot(self.short_trace).shape == (18, 2)

    def test_posteriorplot(self):
        # posteriorplot(self.df_trace).shape == (1,)
        assert posteriorplot(self.short_trace).shape == (18,)

    def test_autocorrplot(self):
        for obj in (self.short_trace, self.fit):
            assert autocorrplot(obj).get_geometry() == (6, 6, 36)

    def test_forestplot(self):
        for obj in (self.short_trace, self.fit, [self.short_trace, self.fit]):
            _, axes = forestplot(obj)
            assert axes.shape == (3,)
            _, axes = forestplot(obj, r_hat=False)
            assert axes.shape == (2,)
            _, axes = forestplot(obj, n_eff=False)
            assert axes.shape == (2,)
            _, axes = forestplot(obj, joyplot=True, r_hat=False, n_eff=False)
            assert axes.shape == (1,)

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
        ppcplot(self.data['y'], self.sample_ppc)

    def test_violintraceplot(self):
        violintraceplot(self.df_trace)
        violintraceplot(self.short_trace)
