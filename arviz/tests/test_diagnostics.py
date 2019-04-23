"""Test Diagnostic methods"""
# pylint: disable=redefined-outer-name, no-member
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from ..data import load_arviz_data
from ..stats import bfmi, rhat, effective_sample_size, mcse, geweke
from ..stats.diagnostics import ks_summary, _multichain_statistics, _mc_error, _rhat_rank_normalized

# For tests only, recommended value should be closer to 1.01-1.05
# See discussion in https://github.com/stan-dev/rstan/pull/618
GOOD_RHAT = 1.1


@pytest.fixture(scope="session")
def data():
    centered_eight = load_arviz_data("centered_eight")
    return centered_eight.posterior


class TestDiagnostics:
    def test_bfmi(self):
        energy = np.array([1, 2, 3, 4])
        assert_almost_equal(bfmi(energy), 0.8)

    @pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
    def test_rhat(self, data, var_names):
        """Confirm rank normalized Split R-hat statistic is close to 1 for a large
        number of samples. Also checks the correct shape"""
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
        ("bulk", "tail", "quantile", "mean", "sd", "median", "mad", "z_scale", "folded", "split"),
    )
    @pytest.mark.parametrize("relative", (True, False))
    def test_effective_sample_size_array(self, ess, relative):
        n_low = 100 if not relative else 100 / 400
        n_high = 800 if not relative else 800 / 400
        if ess in ("quantile", "tail"):
            ess_hat = effective_sample_size(
                np.random.randn(4, 100), method=ess, prob=0.34, relative=relative
            )
            if ess == "tail":
                assert ess_hat > n_low
                assert ess_hat < n_high
                ess_hat = effective_sample_size(
                    np.random.randn(4, 100), method=ess, relative=relative
                )
                assert ess_hat > n_low
                assert ess_hat < n_high
                ess_hat = effective_sample_size(
                    np.random.randn(4, 100), method=ess, prob=(0.2, 0.8), relative=relative
                )
        else:
            ess_hat = effective_sample_size(np.random.randn(4, 100), relative=relative)
        assert ess_hat > n_low
        assert ess_hat < n_high

    @pytest.mark.parametrize(
        "ess",
        ("bulk", "tail", "quantile", "mean", "sd", "median", "mad", "z_scale", "folded", "split"),
    )
    @pytest.mark.parametrize("relative", (True, False))
    def test_effective_sample_size_bad_shape(self, ess, relative):
        with pytest.raises(TypeError):
            if ess in ("quantile", "tail"):
                effective_sample_size(np.random.randn(3), method=ess, prob=0.34, relative=relative)
            else:
                effective_sample_size(np.random.randn(3), method=ess, relative=relative)

    @pytest.mark.parametrize("relative", (True, False))
    def test_effective_sample_size_missing_prob(self, relative):
        with pytest.raises(TypeError):
            effective_sample_size(np.random.randn(4, 100), method="quantile", relative=relative)

    @pytest.mark.parametrize(
        "ess",
        ("bulk", "tail", "quantile", "mean", "sd", "median", "mad", "z_scale", "folded", "split"),
    )
    @pytest.mark.parametrize("relative", (True, False))
    def test_effective_sample_size_bad_chains(self, ess, relative):
        with pytest.raises(TypeError):
            if ess in ("quantile", "tail"):
                effective_sample_size(
                    np.random.randn(1, 3), method=ess, prob=0.34, relative=relative
                )
            else:
                effective_sample_size(np.random.randn(1, 3), method=ess, relative=relative)

    @pytest.mark.parametrize(
        "ess",
        ("bulk", "tail", "quantile", "mean", "sd", "median", "mad", "z_scale", "folded", "split"),
    )
    @pytest.mark.parametrize("relative", (True, False))
    @pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
    def test_effective_sample_size_dataset(self, data, ess, var_names, relative):
        n_low = 100 if not relative else 100 / (data.chain.size * data.draw.size)
        if ess in ("quantile", "tail"):
            ess_hat = effective_sample_size(
                data, var_names=var_names, method=ess, prob=0.34, relative=relative
            )
        else:
            ess_hat = effective_sample_size(
                data, var_names=var_names, method=ess, relative=relative
            )
        assert np.all(ess_hat.mu.values > n_low)  # This might break if the data is regenerated

    @pytest.mark.parametrize("mcse_method", ("mean", "sd", "quantile"))
    def test_mcse_array(self, mcse_method):
        if mcse_method == "quantile":
            mcse_hat = mcse(np.random.randn(4, 100), method=mcse_method, prob=0.34)
        else:
            mcse_hat = mcse(np.random.randn(4, 100), method=mcse_method)
        assert mcse_hat

    @pytest.mark.parametrize("mcse_method", ("mean", "sd", "quantile"))
    @pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
    def test_mcse_dataset(self, data, mcse_method, var_names):
        if mcse_method == "quantile":
            mcse_hat = mcse(data, var_names=var_names, method=mcse_method, prob=0.34)
        else:
            mcse_hat = mcse(data, var_names=var_names, method=mcse_method)
        assert mcse_hat  # This might break if the data is regenerated

    def test_multichain_summary_array(self):
        """Test multichain statistics against invidual functions."""
        ary = np.random.randn(4, 100)
        mcse_mean_hat = mcse(ary, method="mean")
        mcse_sd_hat = mcse(ary, method="sd")
        ess_mean_hat = effective_sample_size(ary, method="mean")
        ess_sd_hat = effective_sample_size(ary, method="sd")
        ess_bulk_hat = effective_sample_size(ary, method="bulk")
        ess_tail_hat = effective_sample_size(ary, method="tail")
        rhat_hat = _rhat_rank_normalized(ary)
        (
            mcse_mean_hat_,
            mcse_sd_hat_,
            ess_mean_hat_,
            ess_sd_hat_,
            ess_bulk_hat_,
            ess_tail_hat_,
            rhat_hat_,
        ) = _multichain_statistics(ary)
        assert mcse_mean_hat == mcse_mean_hat_
        assert mcse_sd_hat == mcse_sd_hat_
        assert ess_mean_hat == ess_mean_hat_
        assert ess_sd_hat == ess_sd_hat_
        assert ess_bulk_hat == ess_bulk_hat_
        assert ess_tail_hat == ess_tail_hat_
        assert round(rhat_hat, 3) == round(rhat_hat_, 3)

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

    @pytest.mark.parametrize("size", [100, 101])
    @pytest.mark.parametrize("batches", [1, 2, 3, 5, 7])
    @pytest.mark.parametrize("ndim", [1, 2, 3])
    @pytest.mark.parametrize("circular", [False, True])
    def test_mc_error(self, size, batches, ndim, circular):
        x = np.random.randn(size, ndim).squeeze()  # pylint: disable=no-member
        assert _mc_error(x, batches=batches, circular=circular) is not None
