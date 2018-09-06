import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
import pymc3 as pm
import pytest

from .helpers import eight_schools_params, load_cached_models, BaseArvizTest
from ..plots import (densityplot, traceplot, energyplot, posteriorplot, autocorrplot, forestplot,
                     parallelplot, pairplot, jointplot, ppcplot, violintraceplot)


class SetupPlots(BaseArvizTest):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.data = eight_schools_params()
        models = load_cached_models(draws=500, chains=2)
        model, cls.short_trace = models['pymc3']
        with model:
            cls.sample_ppc = pm.sample_ppc(cls.short_trace, 100)
        cls.stan_model, cls.fit = models['pystan']
        cls.df_trace = DataFrame({'a': np.random.poisson(2.3, 100)})

    def teardown_method(self):
        super().teardown_method()
        plt.close('all')


class TestPlots(SetupPlots):
    def test_density_plot(self):
        for obj in (self.short_trace, self.fit):
            assert densityplot(obj).shape == (18, 1)

    @pytest.mark.parametrize('combined', [True, False])
    def test_traceplot(self, combined):
        for obj in (self.short_trace, self.fit):
            axes = traceplot(obj, var_names=('mu', 'tau'),
                             combined=combined, lines=[('mu', {}, [1, 2])])
            assert axes.shape == (2, 2)

    def test_autocorrplot(self):
        for obj in (self.short_trace, self.fit):
            assert autocorrplot(obj).shape == (1, 36)

    def test_forestplot(self):
        for obj in (self.short_trace, self.fit, [self.short_trace, self.fit]):
            _, axes = forestplot(obj)
            assert axes.shape == (3,)
            _, axes = forestplot(obj, r_hat=False, quartiles=False)
            assert axes.shape == (2,)
            _, axes = forestplot(obj, var_names=['mu'], colors='C0', n_eff=False, combined=True)
            assert axes.shape == (2,)
            _, axes = forestplot(obj, kind='ridgeplot', r_hat=False, n_eff=False)
            assert axes.shape == (1,)

    @pytest.mark.parametrize('kind', ['kde', 'hist'])
    def test_energyplot(self, kind):
        for obj in (self.short_trace, self.fit):
            assert energyplot(obj, kind=kind)

    def test_parallelplot(self):
        with pytest.raises(ValueError):
            parallelplot(self.df_trace)
        assert parallelplot(self.short_trace)

    @pytest.mark.parametrize('kind', ['scatter', 'hexbin'])
    def test_jointplot(self, kind):
        for obj in (self.short_trace, self.fit):
            jointplot(obj, var_names=('mu', 'tau'), kind=kind)

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
        for obj in (self.short_trace, self.fit):
            axes = violintraceplot(obj)
            assert axes.shape == (18,)


class TestPosteriorPlot(SetupPlots):

    def test_posteriorplot(self):
        for obj in (self.short_trace, self.fit):
            axes = posteriorplot(obj, var_names=('mu', 'tau'), rope=(-2, 2), ref_val=0)
            assert axes.shape == (2,)

    @pytest.mark.parametrize("point_estimate", ('mode', 'mean', 'median'))
    def test_point_estimates(self, point_estimate):
        for obj in (self.short_trace, self.fit):
            axes = posteriorplot(obj, var_names=('mu', 'tau'), point_estimate=point_estimate)
            assert axes.shape == (2,)
