from numpy.testing import assert_allclose
from .helpers import eight_schools_params, load_cached_models
from ..stats import gelman_rubin, effective_n, geweke


class SetupPlots():

    @classmethod
    def setup_class(cls):
        cls.data = eight_schools_params()
        models = load_cached_models(draws=500, chains=2)
        _, cls.short_trace = models['pymc3']

class TestDiagnostics(SetupPlots):
    good_rhat = 1.1

    def test_gelman_rubin(self):
        """Confirm Gelman-Rubin statistic is close to 1 for a large number of samples.
        Also checks the correct shape"""
        rhat = gelman_rubin(self.short_trace)
        assert all(1 / self.good_rhat < r < self.good_rhat for r in rhat['r_hat'].values)


    def test_gelman_rubin_bad(self):
        """Confirm Gelman-Rubin statistic is far from 1 for a small number of samples."""
        rhat = gelman_rubin(self.short_trace[:4])
        assert not all(1 / self.good_rhat < r < self.good_rhat for r in rhat['r_hat'].values)

    def test_effective_n(self):
        eff_n = effective_n(self.short_trace)
        assert_allclose(eff_n['n_eff'].values, 500, 2)

    def test_geweke(self):
        gw_stat = geweke(self.short_trace)
        assert max(abs(gw_stat.iloc[0].geweke[:, 1])) < 1
        assert max(abs(gw_stat.iloc[0].geweke[:, 0])) > -1
