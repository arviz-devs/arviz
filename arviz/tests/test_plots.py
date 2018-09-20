import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
import pymc3 as pm
import pytest

from arviz import from_pymc3
from .helpers import eight_schools_params, load_cached_models, BaseArvizTest
from ..plots import (plot_density, plot_trace, plot_energy, plot_posterior,
                     plot_autocorr, plot_forest, plot_parallel, plot_pair,
                     plot_joint, plot_ppc, plot_violin)


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
    def test_plot_density(self):
        for obj in (self.short_trace, self.fit):
            axes = plot_density(obj)
            assert axes.shape[0] >= 18
            assert axes.shape[1] == 1

    @pytest.mark.parametrize('combined', [True, False])
    def test_plot_trace(self, combined):
        for obj in (self.short_trace, self.fit):
            axes = plot_trace(obj, var_names=('mu', 'tau'),
                              combined=combined, lines=[('mu', {}, [1, 2])])
            assert axes.shape == (2, 2)

    def test_plot_autocorr(self):
        for obj in (self.short_trace, self.fit):
            axes = plot_autocorr(obj)
            assert axes.shape[0] == 1
            assert axes.shape[1] >= 36

    def test_plot_forest(self):
        for obj in (self.short_trace, self.fit, [self.short_trace, self.fit]):
            _, axes = plot_forest(obj)
            assert axes.shape == (3,)
            _, axes = plot_forest(obj, r_hat=False, quartiles=False)
            assert axes.shape == (2,)
            _, axes = plot_forest(obj, var_names=['mu'], colors='C0', n_eff=False, combined=True)
            assert axes.shape == (2,)
            _, axes = plot_forest(obj, kind='ridgeplot', r_hat=False, n_eff=False)
            assert axes.shape == (1,)

    @pytest.mark.parametrize('kind', ['kde', 'hist'])
    def test_plot_energy(self, kind):
        for obj in (self.short_trace, self.fit):
            assert plot_energy(obj, kind=kind)

    def test_plot_parallel(self):
        with pytest.raises(ValueError):
            plot_parallel(self.df_trace)
        assert plot_parallel(self.short_trace)

    @pytest.mark.parametrize('kind', ['scatter', 'hexbin'])
    def test_plot_joint(self, kind):
        for obj in (self.short_trace, self.fit):
            plot_joint(obj, var_names=('mu', 'tau'), kind=kind)

    def test_plot_pair(self):

        plot_pair(self.short_trace, var_names=['theta'], divergences=True,
                  coords={'theta_dim_0': [0, 1]}, plot_kwargs={'marker': 'x'},
                  divergences_kwargs={'marker': '*', 'c': 'C'})

        plot_pair(self.short_trace, divergences=True, plot_kwargs={'marker': 'x'},
                  divergences_kwargs={'marker': '*', 'c': 'C'})
        plot_pair(self.short_trace, kind='hexbin', var_names=['theta'],
                  coords={'theta_dim_0': [0, 1]}, plot_kwargs={'cmap': 'viridis'}, textsize=20)

    @pytest.mark.parametrize('kind', ['density', 'cumulative'])
    def test_plot_ppc(self, kind):
        data = from_pymc3(trace=self.short_trace, posterior_predictive=self.sample_ppc)
        plot_ppc(data, kind=kind)

    def test_plot_violin(self):
        for obj in (self.short_trace, self.fit):
            axes = plot_violin(obj)
            assert axes.shape[0] >= 18


class TestPosteriorPlot(SetupPlots):

    def test_plot_posterior(self):
        for obj in (self.short_trace, self.fit):
            axes = plot_posterior(obj, var_names=('mu', 'tau'), rope=(-2, 2), ref_val=0)
            assert axes.shape == (2,)

    @pytest.mark.parametrize("point_estimate", ('mode', 'mean', 'median'))
    def test_point_estimates(self, point_estimate):
        for obj in (self.short_trace, self.fit):
            axes = plot_posterior(obj, var_names=('mu', 'tau'), point_estimate=point_estimate)
            assert axes.shape == (2,)
