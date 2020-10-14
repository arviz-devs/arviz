# pylint: disable=redefined-outer-name
import os
import sys
from glob import glob

import numpy as np
import pytest

from arviz import from_cmdstanpy

from ..helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    importorskip,
    load_cached_models,
    pystan_version,
)


def _create_test_data():
    """Create test data to local folder.

    This function is needed when test data needs to be updated.
    """
    import platform
    import shutil
    from pathlib import Path

    import cmdstanpy

    model_code = """
        data {
            int<lower=0> J;
            real y[J];
            real<lower=0> sigma[J];
        }

        parameters {
            real mu;
            real<lower=0> tau;
            real eta[J];
        }

        transformed parameters {
            real theta[J];
            for (j in 1:J)
                theta[j] = mu + tau * eta[j];
        }

        model {
            mu ~ normal(0, 5);
            tau ~ cauchy(0, 5);
            eta ~ normal(0, 1);
            y ~ normal(theta, sigma);
        }

        generated quantities {
            vector[J] log_lik;
            vector[J] y_hat;
            for (j in 1:J) {
                log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
                y_hat[j] = normal_rng(theta[j], sigma[j]);
            }
        }
    """
    stan_file = "stan_test_data.stan"
    with open(stan_file, "w") as file_handle:
        print(model_code, file=file_handle)
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    os.remove(stan_file)
    stan_data = {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }
    fit_no_warmup = model.sample(
        data=stan_data, iter_sampling=100, iter_warmup=1000, save_warmup=False
    )
    fit_no_warmup.save_csvfiles(dir=".")
    fit_files = {
        "cmdstanpy_eight_schools_nowarmup": [],
        "cmdstanpy_eight_schools_warmup": [],
    }
    for path in fit_no_warmup.runset.csv_files:
        path = Path(path)
        _, num = path.stem.rsplit("-", 1)
        new_path = path.parent / ("cmdstanpy_eight_schools_nowarmup-" + num + path.suffix)
        shutil.move(path, new_path)
        fit_files["cmdstanpy_eight_schools_nowarmup"].append(new_path)
    fit_warmup = model.sample(data=stan_data, iter_sampling=100, iter_warmup=500, save_warmup=True)
    fit_warmup.save_csvfiles(dir=".")
    for path in fit_no_warmup.runset.csv_files:
        path = Path(path)
        _, num = path.stem.rsplit("-", 1)
        new_path = path.parent / ("cmdstanpy_eight_schools_warmup-" + num + path.suffix)
        shutil.move(path, new_path)
        fit_files["cmdstanpy_eight_schools_warmup"].append(new_path)
    path = Path(stan_file)
    os.remove(str(path.parent / (path.stem + (".exe" if platform.system() == "Windows" else ""))))
    os.remove(str(path.parent / (path.stem + ".hpp")))
    return fit_files


@pytest.mark.skipif(sys.version_info < (3, 6), reason="CmdStanPy is supported only Python 3.6+")
class TestDataCmdStanPy:
    @pytest.fixture(scope="session")
    def data_directory(self):
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "..", "saved_models")
        return data_directory

    @pytest.fixture(scope="class")
    def filepaths(self, data_directory):
        files = {
            "nowarmup": glob(
                os.path.join(
                    data_directory, "cmdstanpy", "cmdstanpy_eight_schools_nowarmup-[1-4].csv"
                )
            ),
            "warmup": glob(
                os.path.join(
                    data_directory, "cmdstanpy", "cmdstanpy_eight_schools_warmup-[1-4].csv"
                )
            ),
        }
        return files

    @pytest.fixture(scope="class")
    def data(self, filepaths):
        # Skip tests if cmdstanpy not installed
        cmdstanpy = importorskip("cmdstanpy")
        CmdStanMCMC = cmdstanpy.CmdStanMCMC  # pylint: disable=invalid-name
        RunSet = cmdstanpy.stanfit.RunSet  # pylint: disable=invalid-name
        CmdStanArgs = cmdstanpy.model.CmdStanArgs  # pylint: disable=invalid-name
        SamplerArgs = cmdstanpy.model.SamplerArgs  # pylint: disable=invalid-name

        class Data:
            args = CmdStanArgs(
                "dummy.stan",
                "dummy.exe",
                list(range(1, 5)),
                method_args=SamplerArgs(iter_sampling=100),
            )
            runset_obj = RunSet(args)
            runset_obj._csv_files = filepaths["nowarmup"]  # pylint: disable=protected-access
            obj = CmdStanMCMC(runset_obj)
            obj.validate_csv_files()  # pylint: disable=protected-access
            obj._assemble_draws()  # pylint: disable=protected-access

            args_warmup = CmdStanArgs(
                "dummy.stan",
                "dummy.exe",
                list(range(1, 5)),
                method_args=SamplerArgs(iter_sampling=100, iter_warmup=500, save_warmup=True),
            )
            runset_obj_warmup = RunSet(args_warmup)
            runset_obj_warmup._csv_files = filepaths["warmup"]  # pylint: disable=protected-access
            obj_warmup = CmdStanMCMC(runset_obj_warmup)
            obj_warmup.validate_csv_files()  # pylint: disable=protected-access
            obj_warmup._assemble_draws()  # pylint: disable=protected-access

        return Data

    def get_inference_data(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(
            posterior=data.obj,
            posterior_predictive="y_hat",
            predictions="y_hat",
            prior=data.obj,
            prior_predictive="y_hat",
            observed_data={"y": eight_schools_params["y"]},
            constant_data={"y": eight_schools_params["y"]},
            predictions_constant_data={"y": eight_schools_params["y"]},
            log_likelihood="log_lik",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
            },
        )

    def get_inference_data2(self, data, eight_schools_params):
        """vars as lists."""
        return from_cmdstanpy(
            posterior=data.obj,
            posterior_predictive=["y_hat"],
            predictions=["y_hat", "log_lik"],
            prior=data.obj,
            prior_predictive=["y_hat"],
            observed_data={"y": eight_schools_params["y"]},
            constant_data=eight_schools_params,
            predictions_constant_data=eight_schools_params,
            log_likelihood=["log_lik", "y_hat"],
            coords={
                "school": np.arange(eight_schools_params["J"]),
                "log_lik_dim": np.arange(eight_schools_params["J"]),
            },
            dims={
                "theta": ["school"],
                "y": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
                "log_lik": ["log_lik_dim"],
            },
        )

    def get_inference_data3(self, data, eight_schools_params):
        """multiple vars as lists."""
        return from_cmdstanpy(
            posterior=data.obj,
            posterior_predictive=["y_hat", "log_lik"],
            prior=data.obj,
            prior_predictive=["y_hat", "log_lik"],
            observed_data={"y": eight_schools_params["y"]},
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "y": ["school"], "y_hat": ["school"], "eta": ["school"]},
        )

    def get_inference_data4(self, data, eight_schools_params):
        """multiple vars as lists."""
        return from_cmdstanpy(
            posterior=data.obj,
            posterior_predictive=None,
            prior=data.obj,
            prior_predictive=None,
            observed_data={"y": eight_schools_params["y"]},
            coords=None,
            dims=None,
        )

    def get_inference_data_warmup_true_is_true(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(
            posterior=data.obj_warmup,
            posterior_predictive="y_hat",
            predictions="y_hat",
            prior=data.obj_warmup,
            prior_predictive="y_hat",
            observed_data={"y": eight_schools_params["y"]},
            constant_data={"y": eight_schools_params["y"]},
            predictions_constant_data={"y": eight_schools_params["y"]},
            log_likelihood="log_lik",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
            },
            save_warmup=True,
        )

    def get_inference_data_warmup_false_is_true(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(
            posterior=data.obj,
            posterior_predictive="y_hat",
            predictions="y_hat",
            prior=data.obj,
            prior_predictive="y_hat",
            observed_data={"y": eight_schools_params["y"]},
            constant_data={"y": eight_schools_params["y"]},
            predictions_constant_data={"y": eight_schools_params["y"]},
            log_likelihood="log_lik",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
            },
            save_warmup=True,
        )

    def get_inference_data_warmup_true_is_false(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(
            posterior=data.obj_warmup,
            posterior_predictive="y_hat",
            predictions="y_hat",
            prior=data.obj_warmup,
            prior_predictive="y_hat",
            observed_data={"y": eight_schools_params["y"]},
            constant_data={"y": eight_schools_params["y"]},
            predictions_constant_data={"y": eight_schools_params["y"]},
            log_likelihood="log_lik",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
            },
            save_warmup=False,
        )

    def test_sampler_stats(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        test_dict = {"sample_stats": ["lp", "diverging"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_inference_data(self, data, eight_schools_params):
        inference_data1 = self.get_inference_data(data, eight_schools_params)
        inference_data2 = self.get_inference_data2(data, eight_schools_params)
        inference_data3 = self.get_inference_data3(data, eight_schools_params)
        inference_data4 = self.get_inference_data4(data, eight_schools_params)
        # inference_data 1
        test_dict = {
            "posterior": ["theta"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "constant_data": ["y"],
            "predictions_constant_data": ["y"],
            "log_likelihood": ["log_lik"],
            "prior": ["theta"],
        }
        fails = check_multiple_attrs(test_dict, inference_data1)
        assert not fails
        # inference_data 2
        test_dict = {
            "posterior_predictive": ["y_hat"],
            "predictions": ["y_hat", "log_lik"],
            "observed_data": ["y"],
            "sample_stats_prior": ["lp"],
            "sample_stats": ["lp"],
            "constant_data": list(eight_schools_params),
            "predictions_constant_data": list(eight_schools_params),
            "prior_predictive": ["y_hat"],
            "log_likelihood": ["log_lik", "y_hat"],
        }
        fails = check_multiple_attrs(test_dict, inference_data2)
        assert not fails
        # inference_data 3
        test_dict = {
            "posterior_predictive": ["y_hat"],
            "observed_data": ["y"],
            "sample_stats_prior": ["lp"],
            "sample_stats": ["lp"],
            "prior_predictive": ["y_hat"],
        }
        fails = check_multiple_attrs(test_dict, inference_data3)
        assert not fails
        # inference_data 4
        test_dict = {"posterior": ["theta"], "prior": ["theta"]}
        fails = check_multiple_attrs(test_dict, inference_data4)
        assert not fails

    def test_inference_data_warmup(self, data, eight_schools_params):
        inference_data_true_is_true = self.get_inference_data_warmup_true_is_true(
            data, eight_schools_params
        )
        inference_data_false_is_true = self.get_inference_data_warmup_false_is_true(
            data, eight_schools_params
        )
        inference_data_true_is_false = self.get_inference_data_warmup_true_is_false(
            data, eight_schools_params
        )
        inference_data_false_is_false = self.get_inference_data(data, eight_schools_params)
        # inference_data warmup
        test_dict = {
            "posterior": ["theta"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "constant_data": ["y"],
            "predictions_constant_data": ["y"],
            "log_likelihood": ["log_lik"],
            "prior": ["theta"],
            "warmup_posterior": ["theta"],
            "warmup_predictions": ["y_hat"],
            "warmup_log_likelihood": ["log_lik"],
            "warmup_prior": ["theta"],
        }
        fails = check_multiple_attrs(test_dict, inference_data_true_is_true)
        assert not fails
        # inference_data no warmup
        test_dict = {
            "posterior": ["theta"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "constant_data": ["y"],
            "predictions_constant_data": ["y"],
            "log_likelihood": ["log_lik"],
            "prior": ["theta"],
        }
        fails = check_multiple_attrs(test_dict, inference_data_false_is_true)
        assert not fails
        assert "warmup_posterior" not in inference_data_false_is_true
        assert "warmup_predictions" not in inference_data_false_is_true
        assert "warmup_log_likelihood" not in inference_data_false_is_true
        assert "warmup_prior" not in inference_data_false_is_true
        # inference_data no warmup
        test_dict = {
            "posterior": ["theta"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "constant_data": ["y"],
            "predictions_constant_data": ["y"],
            "log_likelihood": ["log_lik"],
            "prior": ["theta"],
        }
        fails = check_multiple_attrs(test_dict, inference_data_true_is_false)
        assert not fails
        assert "warmup_posterior" not in inference_data_true_is_false
        assert "warmup_predictions" not in inference_data_true_is_false
        assert "warmup_log_likelihood" not in inference_data_true_is_false
        assert "warmup_prior" not in inference_data_true_is_false
        # inference_data no warmup
        test_dict = {
            "posterior": ["theta"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "constant_data": ["y"],
            "predictions_constant_data": ["y"],
            "log_likelihood": ["log_lik"],
            "prior": ["theta"],
        }
        fails = check_multiple_attrs(test_dict, inference_data_false_is_false)
        assert not fails
        assert "warmup_posterior" not in inference_data_false_is_false
        assert "warmup_predictions" not in inference_data_false_is_false
        assert "warmup_log_likelihood" not in inference_data_false_is_false
        assert "warmup_prior" not in inference_data_false_is_false
