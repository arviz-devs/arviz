import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

from ..stats import gelman_rubin, effective_n, geweke


class TestDiagnostics(object):
    good_rhat = 1.1

    def fake_trace(self, n_samples):
        """
        Creates a fake trace with 3 variables and 2 chains
        """
        alpha = np.repeat((1, 5, 10), n_samples)
        beta = np.repeat((1, 5, 1), n_samples)
        data = np.random.beta(alpha, beta).reshape(-1, n_samples//2)
        trace = pd.DataFrame(data.T, columns=['a', 'a', 'b', 'b', 'c', 'c'])
        return trace

    def test_gelman_rubin(self):
        """Confirm Gelman-Rubin statistic is close to 1 for a large number of samples.
        Also checks the correct shape"""
        trace = self.fake_trace(1000)
        rhat = gelman_rubin(trace)
        assert all(1 / self.good_rhat < r < self.good_rhat for r in rhat.values)
        assert rhat.shape == (3,)

    def test_gelman_rubin_bad(self):
        """Confirm Gelman-Rubin statistic is far from 1 for a small number of samples."""
        trace = self.fake_trace(6)
        rhat = gelman_rubin(trace)
        assert not all(1 / self.good_rhat < r < self.good_rhat for r in rhat.values)

    def test_effective_n(self):
        n_samples = 1000
        trace = self.fake_trace(n_samples)
        eff_n = effective_n(trace)
        assert_allclose(eff_n, n_samples, 2)
        assert eff_n.shape == (3,)

    def test_geweke(self):
        trace = self.fake_trace(1000)
        gw_stat = geweke(trace)
        assert max(abs(gw_stat['a'][:, 1])) < 1
        assert max(abs(gw_stat['a'][:, 0])) > -1
