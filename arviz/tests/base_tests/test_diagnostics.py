"""Test Diagnostic methods"""
# pylint: disable=redefined-outer-name, no-member, too-many-public-methods
import os

import numpy as np
import packaging
import pandas as pd
import pytest
import scipy
from numpy.testing import assert_almost_equal

from ...data import from_cmdstan, load_arviz_data
from ...rcparams import rcParams
from ...sel_utils import xarray_var_iter
from ...stats import bfmi, ess, mcse, rhat
from ...stats.diagnostics import (
    _ess,
    _ess_quantile,
    _mc_error,
    _mcse_quantile,
    _multichain_statistics,
    _rhat,
    _rhat_rank,
    _split_chains,
    _z_scale,
    ks_summary,
)

# For tests only, recommended value should be closer to 1.01-1.05
# See discussion in https://github.com/stan-dev/rstan/pull/618
GOOD_RHAT = 1.1

rcParams["data.load"] = "eager"


@pytest.fixture(scope="session")
def data():
    centered_eight = load_arviz_data("centered_eight")
    return centered_eight.posterior


class TestDiagnostics:
    def test_bfmi(self):
        energy = np.array([1, 2, 3, 4])
        assert_almost_equal(bfmi(energy), 0.6)

    def test_bfmi_dataset(self):
        data = load_arviz_data("centered_eight")
        assert bfmi(data).all()

    def test_bfmi_dataset_bad(self):
        data = load_arviz_data("centered_eight")
        del data.sample_stats["energy"]
        with pytest.raises(TypeError):
            bfmi(data)

    def test_bfmi_correctly_transposed(self):
        data = load_arviz_data("centered_eight")
        vals1 = bfmi(data)
        data.sample_stats["energy"] = data.sample_stats["energy"].T
        vals2 = bfmi(data)
        assert_almost_equal(vals1, vals2)

    def test_deterministic(self):
        """
        Test algorithm against posterior (R) convergence functions.

        posterior: https://github.com/stan-dev/posterior
        R code:
        ```
        library("posterior")
        data2 <- read.csv("blocker.2.csv", comment.char = "#")
        data1 <- read.csv("blocker.1.csv", comment.char = "#")
        output <- matrix(ncol=17, nrow=length(names(data1))-4)
        j = 0
        for (i in 1:length(names(data1))) {
            name = names(data1)[i]
            ary = matrix(c(data1[,name], data2[,name]), 1000, 2)
            if (!endsWith(name, "__"))
                j <- j + 1
                output[j,] <- c(
                    posterior::rhat(ary),
                    posterior::rhat_basic(ary, FALSE),
                    posterior::ess_bulk(ary),
                    posterior::ess_tail(ary),
                    posterior::ess_mean(ary),
                    posterior::ess_sd(ary),
                    posterior::ess_median(ary),
                    posterior::ess_basic(ary, FALSE),
                    posterior::ess_quantile(ary, 0.01),
                    posterior::ess_quantile(ary, 0.1),
                    posterior::ess_quantile(ary, 0.3),
                    posterior::mcse_mean(ary),
                    posterior::mcse_sd(ary),
                    posterior::mcse_median(ary),
                    posterior::mcse_quantile(ary, prob=0.01),
                    posterior::mcse_quantile(ary, prob=0.1),
                    posterior::mcse_quantile(ary, prob=0.3))
        }
        df = data.frame(output, row.names = names(data1)[5:ncol(data1)])
        colnames(df) <- c("rhat_rank",
                          "rhat_raw",
                          "ess_bulk",
                          "ess_tail",
                          "ess_mean",
                          "ess_sd",
                          "ess_median",
                          "ess_raw",
                          "ess_quantile01",
                          "ess_quantile10",
                          "ess_quantile30",
                          "mcse_mean",
                          "mcse_sd",
                          "mcse_median",
                          "mcse_quantile01",
                          "mcse_quantile10",
                          "mcse_quantile30")
        write.csv(df, "reference_posterior.csv")
        ```
        Reference file:

        Created: 2020-08-31
        System: Ubuntu 18.04.5 LTS
        R version 4.0.2 (2020-06-22)
        posterior 0.1.2
        """
        # download input files
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "..", "saved_models")
        path = os.path.join(data_directory, "stan_diagnostics", "blocker.[0-9].csv")
        posterior = from_cmdstan(path)
        reference_path = os.path.join(data_directory, "stan_diagnostics", "reference_posterior.csv")
        reference = (
            pd.read_csv(reference_path, index_col=0, float_precision="high")
            .sort_index(axis=1)
            .sort_index(axis=0)
        )
        # test arviz functions
        funcs = {
            "rhat_rank": lambda x: rhat(x, method="rank"),
            "rhat_raw": lambda x: rhat(x, method="identity"),
            "ess_bulk": lambda x: ess(x, method="bulk"),
            "ess_tail": lambda x: ess(x, method="tail"),
            "ess_mean": lambda x: ess(x, method="mean"),
            "ess_sd": lambda x: ess(x, method="sd"),
            "ess_median": lambda x: ess(x, method="median"),
            "ess_raw": lambda x: ess(x, method="identity"),
            "ess_quantile01": lambda x: ess(x, method="quantile", prob=0.01),
            "ess_quantile10": lambda x: ess(x, method="quantile", prob=0.1),
            "ess_quantile30": lambda x: ess(x, method="quantile", prob=0.3),
            "mcse_mean": lambda x: mcse(x, method="mean"),
            "mcse_sd": lambda x: mcse(x, method="sd"),
            "mcse_median": lambda x: mcse(x, method="median"),
            "mcse_quantile01": lambda x: mcse(x, method="quantile", prob=0.01),
            "mcse_quantile10": lambda x: mcse(x, method="quantile", prob=0.1),
            "mcse_quantile30": lambda x: mcse(x, method="quantile", prob=0.3),
        }
        results = {}
        for key, coord_dict, _, vals in xarray_var_iter(posterior.posterior, combined=True):
            if coord_dict:
                key = f"{key}.{list(coord_dict.values())[0] + 1}"
            results[key] = {func_name: func(vals) for func_name, func in funcs.items()}
        arviz_data = pd.DataFrame.from_dict(results).T.sort_index(axis=1).sort_index(axis=0)

        # check column names
        assert set(arviz_data.columns) == set(reference.columns)

        # check parameter names
        assert set(arviz_data.index) == set(reference.index)

        # show print with pytests '-s' tag
        np.set_printoptions(16)
        print(abs(reference - arviz_data).max())

        # test absolute accuracy
        assert (abs(reference - arviz_data).values < 1e-8).all(None)

    @pytest.mark.parametrize("method", ("rank", "split", "folded", "z_scale", "identity"))
    @pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
    def test_rhat(self, data, var_names, method):
        """Confirm R-hat statistic is close to 1 for a large
        number of samples. Also checks the correct shape"""
        rhat_data = rhat(data, var_names=var_names, method=method)
        for r_hat in rhat_data.data_vars.values():
            assert ((1 / GOOD_RHAT < r_hat.values) | (r_hat.values < GOOD_RHAT)).all()

        # In None case check that all varnames from rhat_data match input data
        if var_names is None:
            assert list(rhat_data.data_vars) == list(data.data_vars)

    @pytest.mark.parametrize("method", ("rank", "split", "folded", "z_scale", "identity"))
    def test_rhat_nan(self, method):
        """Confirm R-hat statistic returns nan."""
        data = np.random.randn(4, 100)
        data[0, 0] = np.nan  #  pylint: disable=unsupported-assignment-operation
        rhat_data = rhat(data, method=method)
        assert np.isnan(rhat_data)
        if method == "rank":
            assert np.isnan(_rhat(rhat_data))

    @pytest.mark.parametrize("method", ("rank", "split", "folded", "z_scale", "identity"))
    @pytest.mark.parametrize("chain", (None, 1, 2))
    @pytest.mark.parametrize("draw", (1, 2, 3, 4))
    def test_rhat_shape(self, method, chain, draw):
        """Confirm R-hat statistic returns nan."""
        data = np.random.randn(draw) if chain is None else np.random.randn(chain, draw)
        if (chain in (None, 1)) or (draw < 4):
            rhat_data = rhat(data, method=method)
            assert np.isnan(rhat_data)
        else:
            rhat_data = rhat(data, method=method)
            assert not np.isnan(rhat_data)

    def test_rhat_bad(self):
        """Confirm rank normalized Split R-hat statistic is
        far from 1 for a small number of samples."""
        r_hat = rhat(np.vstack([20 + np.random.randn(1, 100), np.random.randn(1, 100)]))
        assert 1 / GOOD_RHAT > r_hat or GOOD_RHAT < r_hat

    def test_rhat_bad_method(self):
        with pytest.raises(TypeError):
            rhat(np.random.randn(2, 300), method="wrong_method")

    def test_rhat_ndarray(self):
        with pytest.raises(TypeError):
            rhat(np.random.randn(2, 300, 10))

    @pytest.mark.parametrize(
        "method",
        (
            "bulk",
            "tail",
            "quantile",
            "local",
            "mean",
            "sd",
            "median",
            "mad",
            "z_scale",
            "folded",
            "identity",
        ),
    )
    @pytest.mark.parametrize("relative", (True, False))
    def test_effective_sample_size_array(self, data, method, relative):
        n_low = 100 / 400 if relative else 100
        n_high = 800 / 400 if relative else 800
        if method in ("quantile", "tail"):
            ess_hat = ess(data, method=method, prob=0.34, relative=relative)
            if method == "tail":
                assert ess_hat > n_low
                assert ess_hat < n_high
                ess_hat = ess(np.random.randn(4, 100), method=method, relative=relative)
                assert ess_hat > n_low
                assert ess_hat < n_high
                ess_hat = ess(
                    np.random.randn(4, 100), method=method, prob=(0.2, 0.8), relative=relative
                )
        elif method == "local":
            ess_hat = ess(
                np.random.randn(4, 100), method=method, prob=(0.2, 0.3), relative=relative
            )
        else:
            ess_hat = ess(np.random.randn(4, 100), method=method, relative=relative)
        assert ess_hat > n_low
        assert ess_hat < n_high

    @pytest.mark.parametrize(
        "method",
        (
            "bulk",
            "tail",
            "quantile",
            "local",
            "mean",
            "sd",
            "median",
            "mad",
            "z_scale",
            "folded",
            "identity",
        ),
    )
    @pytest.mark.parametrize("relative", (True, False))
    @pytest.mark.parametrize("chain", (None, 1, 2))
    @pytest.mark.parametrize("draw", (1, 2, 3, 4))
    @pytest.mark.parametrize("use_nan", (True, False))
    def test_effective_sample_size_nan(self, method, relative, chain, draw, use_nan):
        data = np.random.randn(draw) if chain is None else np.random.randn(chain, draw)
        if use_nan:
            data[0] = np.nan
        if method in ("quantile", "tail"):
            ess_value = ess(data, method=method, prob=0.34, relative=relative)
        elif method == "local":
            ess_value = ess(data, method=method, prob=(0.2, 0.3), relative=relative)
        else:
            ess_value = ess(data, method=method, relative=relative)
        if (draw < 4) or use_nan:
            assert np.isnan(ess_value)
        else:
            assert not np.isnan(ess_value)
        # test following only once tests are run
        if (method == "bulk") and (not relative) and (chain is None) and (draw == 4):
            if use_nan:
                assert np.isnan(_ess(data))
            else:
                assert not np.isnan(_ess(data))

    @pytest.mark.parametrize("relative", (True, False))
    def test_effective_sample_size_missing_prob(self, relative):
        with pytest.raises(TypeError):
            ess(np.random.randn(4, 100), method="quantile", relative=relative)
        with pytest.raises(TypeError):
            _ess_quantile(np.random.randn(4, 100), prob=None, relative=relative)
        with pytest.raises(TypeError):
            ess(np.random.randn(4, 100), method="local", relative=relative)

    @pytest.mark.parametrize("relative", (True, False))
    def test_effective_sample_size_too_many_probs(self, relative):
        with pytest.raises(ValueError):
            ess(np.random.randn(4, 100), method="local", prob=[0.1, 0.2, 0.9], relative=relative)

    def test_effective_sample_size_constant(self):
        assert ess(np.ones((4, 100))) == 400

    def test_effective_sample_size_bad_method(self):
        with pytest.raises(TypeError):
            ess(np.random.randn(4, 100), method="wrong_method")

    def test_effective_sample_size_ndarray(self):
        with pytest.raises(TypeError):
            ess(np.random.randn(2, 300, 10))

    @pytest.mark.parametrize(
        "method",
        (
            "bulk",
            "tail",
            "quantile",
            "local",
            "mean",
            "sd",
            "median",
            "mad",
            "z_scale",
            "folded",
            "identity",
        ),
    )
    @pytest.mark.parametrize("relative", (True, False))
    @pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
    def test_effective_sample_size_dataset(self, data, method, var_names, relative):
        n_low = 100 / (data.chain.size * data.draw.size) if relative else 100
        if method in ("quantile", "tail"):
            ess_hat = ess(data, var_names=var_names, method=method, prob=0.34, relative=relative)
        elif method == "local":
            ess_hat = ess(
                data, var_names=var_names, method=method, prob=(0.2, 0.3), relative=relative
            )
        else:
            ess_hat = ess(data, var_names=var_names, method=method, relative=relative)
        assert np.all(ess_hat.mu.values > n_low)  # This might break if the data is regenerated

    @pytest.mark.parametrize("mcse_method", ("mean", "sd", "median", "quantile"))
    def test_mcse_array(self, mcse_method):
        if mcse_method == "quantile":
            mcse_hat = mcse(np.random.randn(4, 100), method=mcse_method, prob=0.34)
        else:
            mcse_hat = mcse(np.random.randn(4, 100), method=mcse_method)
        assert mcse_hat

    def test_mcse_ndarray(self):
        with pytest.raises(TypeError):
            mcse(np.random.randn(2, 300, 10))

    @pytest.mark.parametrize("mcse_method", ("mean", "sd", "median", "quantile"))
    @pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
    def test_mcse_dataset(self, data, mcse_method, var_names):
        if mcse_method == "quantile":
            mcse_hat = mcse(data, var_names=var_names, method=mcse_method, prob=0.34)
        else:
            mcse_hat = mcse(data, var_names=var_names, method=mcse_method)
        assert mcse_hat  # This might break if the data is regenerated

    @pytest.mark.parametrize("mcse_method", ("mean", "sd", "median", "quantile"))
    @pytest.mark.parametrize("chain", (None, 1, 2))
    @pytest.mark.parametrize("draw", (1, 2, 3, 4))
    @pytest.mark.parametrize("use_nan", (True, False))
    def test_mcse_nan(self, mcse_method, chain, draw, use_nan):
        data = np.random.randn(draw) if chain is None else np.random.randn(chain, draw)
        if use_nan:
            data[0] = np.nan
        if mcse_method == "quantile":
            mcse_hat = mcse(data, method=mcse_method, prob=0.34)
        else:
            mcse_hat = mcse(data, method=mcse_method)
        if draw < 4 or use_nan:
            assert np.isnan(mcse_hat)
        else:
            assert not np.isnan(mcse_hat)

    @pytest.mark.parametrize("method", ("wrong_method", "quantile"))
    def test_mcse_bad_method(self, data, method):
        with pytest.raises(TypeError):
            mcse(data, method=method, prob=None)

    @pytest.mark.parametrize("draws", (3, 4, 100))
    @pytest.mark.parametrize("chains", (None, 1, 2))
    def test_multichain_summary_array(self, draws, chains):
        """Test multichain statistics against individual functions."""
        if chains is None:
            ary = np.random.randn(draws)
        else:
            ary = np.random.randn(chains, draws)

        mcse_mean_hat = mcse(ary, method="mean")
        mcse_sd_hat = mcse(ary, method="sd")
        ess_bulk_hat = ess(ary, method="bulk")
        ess_tail_hat = ess(ary, method="tail")
        rhat_hat = _rhat_rank(ary)
        (
            mcse_mean_hat_,
            mcse_sd_hat_,
            ess_bulk_hat_,
            ess_tail_hat_,
            rhat_hat_,
        ) = _multichain_statistics(ary)
        if draws == 3:
            assert np.isnan(
                (
                    mcse_mean_hat,
                    mcse_sd_hat,
                    ess_bulk_hat,
                    ess_tail_hat,
                    rhat_hat,
                )
            ).all()
            assert np.isnan(
                (
                    mcse_mean_hat_,
                    mcse_sd_hat_,
                    ess_bulk_hat_,
                    ess_tail_hat_,
                    rhat_hat_,
                )
            ).all()
        else:
            assert_almost_equal(mcse_mean_hat, mcse_mean_hat_)
            assert_almost_equal(mcse_sd_hat, mcse_sd_hat_)
            assert_almost_equal(ess_bulk_hat, ess_bulk_hat_)
            assert_almost_equal(ess_tail_hat, ess_tail_hat_)
            if chains in (None, 1):
                assert np.isnan(rhat_hat)
                assert np.isnan(rhat_hat_)
            else:
                assert round(rhat_hat, 3) == round(rhat_hat_, 3)

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

    @pytest.mark.parametrize("size", [100, 101])
    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_mc_error_nan(self, size, ndim):
        x = np.random.randn(size, ndim).squeeze()  # pylint: disable=no-member
        x[0] = np.nan
        if ndim != 1:
            assert np.isnan(_mc_error(x)).all()
        else:
            assert np.isnan(_mc_error(x))

    @pytest.mark.parametrize("func", ("_mcse_quantile", "_z_scale"))
    def test_nan_behaviour(self, func):
        data = np.random.randn(100, 4)
        data[0, 0] = np.nan  #  pylint: disable=unsupported-assignment-operation
        if func == "_mcse_quantile":
            assert np.isnan(_mcse_quantile(data, 0.5)).all(None)
        elif packaging.version.parse(scipy.__version__) < packaging.version.parse("1.10.0.dev0"):
            assert not np.isnan(_z_scale(data)).all(None)
            assert not np.isnan(_z_scale(data)).any(None)
        else:
            assert np.isnan(_z_scale(data)).sum() == 1

    @pytest.mark.parametrize("chains", (None, 1, 2, 3))
    @pytest.mark.parametrize("draws", (2, 3, 100, 101))
    def test_split_chain_dims(self, chains, draws):
        if chains is None:
            data = np.random.randn(draws)
        else:
            data = np.random.randn(chains, draws)
        split_data = _split_chains(data)
        if chains is None:
            chains = 1
        assert split_data.shape == (chains * 2, draws // 2)
