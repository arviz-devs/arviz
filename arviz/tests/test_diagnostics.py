"""Test Diagnostic methods"""
# pylint: disable=redefined-outer-name, no-member
import inspect
import numpy as np
import pytest

from ..data import load_arviz_data
from ..stats import (
    rhat,
    effective_sample_size_mean,
    effective_sample_size_sd,
    effective_sample_size_bulk,
    effective_sample_size_tail,
    effective_sample_size_quantile,
    mcse_mean,
    mcse_sd,
    mcse_quantile,
    geweke,
)
from ..stats.diagnostics import mcse_mean_sd, ks_summary, _multichain_statistics

# For tests only, recommended value should be closer to 1.01
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
        """Confirm rank normalized Split R-hat statistic is
        far from 1 for a small number of samples."""
        r_hat = rhat(np.vstack([20 + np.random.randn(1, 100), np.random.randn(1, 100)]))
        assert 1 / GOOD_RHAT > r_hat or GOOD_RHAT < r_hat

    def test_rhat_bad_shape(self):
        with pytest.raises(TypeError):
            rhat(np.random.randn(3))

    @pytest.mark.parametrize(
        "ess",
        (
            effective_sample_size_mean,
            effective_sample_size_sd,
            effective_sample_size_bulk,
            effective_sample_size_tail,
            effective_sample_size_quantile,
        ),
    )
    def test_effective_sample_size_array(self, ess):
        parameters = list(inspect(ess).patameters.keys())
        if "prob" in parameters:
            ess_hat = ess(np.random.randn(4, 100), prob=0.34)
        else:
            ess_hat = ess(np.random.randn(4, 100))
        assert ess_hat > 100
        assert ess_hat < 800

    @pytest.mark.parametrize(
        "ess",
        (
            effective_sample_size_mean,
            effective_sample_size_sd,
            effective_sample_size_bulk,
            effective_sample_size_tail,
            effective_sample_size_quantile,
        ),
    )
    def test_effective_sample_size_bad_shape(self, ess):
        with pytest.raises(TypeError):
            parameters = list(inspect(ess).patameters.keys())
            if "prob" in parameters:
                effective_sample_size(np.random.randn(3), prob=0.34)
            else:
                effective_sample_size(np.random.randn(3))

    @pytest.mark.parametrize(
        "ess",
        (
            effective_sample_size_mean,
            effective_sample_size_sd,
            effective_sample_size_bulk,
            effective_sample_size_tail,
            effective_sample_size_quantile,
        ),
    )
    def test_effective_sample_size_bad_chains(self, ess):
        with pytest.raises(TypeError):
            parameters = list(inspect(ess).patameters.keys())
            if "prob" in parameters:
                effective_sample_size(np.random.randn(1, 3), prob=0.34)
            else:
                effective_sample_size(np.random.randn(1, 3))

    @pytest.mark.parametrize(
        "ess",
        (
            effective_sample_size_mean,
            effective_sample_size_sd,
            effective_sample_size_bulk,
            effective_sample_size_tail,
            effective_sample_size_quantile,
        ),
    )
    @pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
    def test_effective_sample_size_dataset(self, data, ess, var_names):
        parameters = list(inspect(ess).patameters.keys())
        if "prob" in parameters:
            ess_hat = ess(data, var_names=var_names, prob=0.34)
        else:
            ess_hat = ess(data, var_names=var_names)
        assert ess_hat.mu > 100  # This might break if the data is regenerated

    @pytest.mark.parametrize("mcse", (mcse_mean, mcse_sd, mcse_quantile))
    def test_mcse_array(self, mcse):
        parameters = list(inspect(ess).patameters.keys())
        if "prob" in parameters:
            mcse_hat = mcse(np.random.randn(4, 100), prob=0.34)
        else:
            mcse_hat = mcse(np.random.randn(4, 100))
        assert mcse_hat

    def test_summary_array(self):
        ary = np.random.randn(4, 100)
        mcse_mean = mcse_mean(ary, prob=0.34)
        mcse_sd = _mcse_sd(ary)
        ess_mean = _ess_mean(ary)
        ess_sd = _ess_sd(ary)
        ess_bulk = _ess_bulk(ary)
        ess_tail = _ess_tail(ary)
        rhat = _rhat_rank_normalized(ary)
        mcse_mean_, mcse_sd_, ess_mean_, ess_sd_, ess_bulk_, ess_tail_, rhat_ = _multichain_statistics(
            ary
        )
        assert mcse_mean == mcse_mean_
        assert mcse_sd == mcse_sd_
        assert ess_mean == ess_mean_
        assert ess_sd == ess_sd_
        assert ess_bulk == ess_bulk_
        assert ess_tail == ess_tail_
        assert rhat == rhat_

    @pytest.mark.parametrize("mcse", (mcse_mean, mcse_sd, mcse_quantile))
    @pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
    def test_mcse_dataset(self, data, mcse, var_names):
        parameters = list(inspect(ess).patameters.keys())
        if "prob" in parameters:
            mcse_hat = mcse(data, var_names=var_names, prob=0.34)
        else:
            mcse_hat = mcse(data, var_names=var_names)
        assert mcse_hat  # This might break if the data is regenerated

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
