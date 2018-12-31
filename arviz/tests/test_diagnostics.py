"""Test Diagnostic methods"""
# pylint: disable=redefined-outer-name, no-member
import numpy as np
import pytest

from ..data import load_arviz_data
from ..stats import rhat, effective_n, geweke

GOOD_RHAT = 1.1


@pytest.fixture(scope="session")
def data():
    centered_eight = load_arviz_data("centered_eight")
    return centered_eight.posterior


class TestDiagnostics:
    @pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
    def test_rhat(self, data, var_names):
        """Confirm Split R-hat statistic is close to 1 for a large number of samples.
        Also checks the correct shape"""
        rhat_data = rhat(data, var_names=var_names)
        for r_hat in rhat_data.data_vars.values():
            assert ((1 / GOOD_RHAT < r_hat.values) | (r_hat.values < GOOD_RHAT)).all()

        # In None case check that all varnames from rhat_data match input data
        if var_names is None:
            assert list(rhat_data.data_vars) == list(data.data_vars)

    def test_rhat_bad(self):
        """Confirm Split R-hat statistic is far from 1 for a small number of samples."""
        r_hat = rhat(np.hstack([20 + np.random.randn(100, 1), np.random.randn(100, 1)]))
        assert 1 / GOOD_RHAT > r_hat or GOOD_RHAT < r_hat

    def test_effective_n_array(self):
        eff_n = effective_n(np.random.randn(4, 100))
        assert eff_n > 100
        assert eff_n < 800

    @pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
    def test_effective_n_dataset(self, data, var_names):
        eff_n = effective_n(data, var_names=var_names)
        assert eff_n.mu > 100  # This might break if the data is regenerated

    def test_geweke(self):
        first = 0.1
        last = 0.5
        intervals = 100

        gw_stat = geweke(np.random.randn(10000), first=first, last=last, intervals=intervals)

        # all geweke values should be between -1 and 1 for this many draws from a
        # normal distribution
        assert ((gw_stat[:, 1] > -1) | (gw_stat[:, 1] < 1)).all()

        assert gw_stat.shape[0] == intervals
        assert 10000 * last - gw_stat[:, 0].max() == 1
