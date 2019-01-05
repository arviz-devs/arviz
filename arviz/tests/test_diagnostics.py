"""Test Diagnostic methods"""
# pylint: disable=redefined-outer-name, no-member
import numpy as np
import pytest

from ..data import load_arviz_data
from ..stats import rhat, effective_sample_size, geweke
from ..stats.diagnostics import ks_summary


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
        r_hat = rhat(np.vstack([20 + np.random.randn(1, 100), np.random.randn(1, 100)]))
        assert 1 / GOOD_RHAT > r_hat or GOOD_RHAT < r_hat

    def test_rhat_bad_shape(self):
        with pytest.raises(TypeError):
            rhat(np.random.randn(3))

    def test_effective_sample_size_array(self):
        eff_n_hat = effective_sample_size(np.random.randn(4, 100))
        assert eff_n_hat > 100
        assert eff_n_hat < 800

    def test_effective_sample_size_bad_shape(self):
        with pytest.raises(TypeError):
            effective_sample_size(np.random.randn(3))

    def test_effective_sample_size_bad_chains(self):
        with pytest.raises(TypeError):
            effective_sample_size(np.random.randn(1, 3))

    @pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
    def test_effective_sample_size_dataset(self, data, var_names):
        eff_n_hat = effective_sample_size(data, var_names=var_names)
        assert eff_n_hat.mu > 100  # This might break if the data is regenerated

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

    def test_geweke_bad_interval(self):
        # lower bound
        with pytest.raises(ValueError):
            geweke(np.random.randn(10), first=0)
        # upper bound
        with pytest.raises(ValueError):
            geweke(np.random.randn(10), last=1)
        # sum larger than 1
        with pytest.raises(ValueError):
            geweke(np.random.randn(10), first=0.9, last=0.9)

    def test_ks_summary(self):
        """Instead of psislw data, this test uses fake data."""
        pareto_tail_indices = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
        with pytest.warns(UserWarning):
            summary = ks_summary(pareto_tail_indices)
        assert summary is not None
        pareto_tail_indices2 = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.6])
        with pytest.warns(UserWarning):
            summary2 = ks_summary(pareto_tail_indices2)
        assert summary2 is not None
