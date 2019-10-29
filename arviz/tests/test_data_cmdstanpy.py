# pylint: disable=redefined-outer-name
from glob import glob
import os
import sys
import numpy as np
import pytest


from arviz import from_cmdstanpy
from .helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    load_cached_models,
    pystan_version,
)


@pytest.mark.skipif(sys.version_info < (3, 6), reason="CmdStanPy is supported only Python 3.6+")
class TestDataCmdStanPy:
    @pytest.fixture(scope="session")
    def data_directory(self):
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "saved_models")
        return data_directory

    @pytest.fixture(scope="class")
    def filepaths(self, data_directory):
        files = glob(os.path.join(data_directory, "cmdstanpy", "cmdstanpy_eight_schools-[1-4].csv"))
        return files

    @pytest.fixture(scope="class")
    def data(self, filepaths):
        from cmdstanpy import CmdStanMCMC
        from cmdstanpy.stanfit import RunSet
        from cmdstanpy.model import CmdStanArgs, SamplerArgs

        class Data:
            args = CmdStanArgs(
                "dummy.stan", "dummy.exe", list(range(1, 5)), method_args=SamplerArgs()
            )
            runset_obj = RunSet(args)
            runset_obj._csv_files = filepaths  # pylint: disable=protected-access
            obj = CmdStanMCMC(runset_obj)
            obj._validate_csv_files()  # pylint: disable=protected-access
            obj._assemble_sample()  # pylint: disable=protected-access

        return Data

    def get_inference_data(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(
            posterior=data.obj,
            posterior_predictive="y_hat",
            prior=data.obj,
            prior_predictive="y_hat",
            observed_data={"y": eight_schools_params["y"]},
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
            prior=data.obj,
            prior_predictive=["y_hat"],
            observed_data={"y": eight_schools_params["y"]},
            log_likelihood="log_lik",
            coords={
                "school": np.arange(eight_schools_params["J"]),
                "log_likelihood_dim": np.arange(eight_schools_params["J"]),
            },
            dims={
                "theta": ["school"],
                "y": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
                "log_lik": ["log_likelihood_dim"],
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
            "observed_data": ["y"],
            "sample_stats": ["log_likelihood"],
            "prior": ["theta"],
        }
        fails = check_multiple_attrs(test_dict, inference_data1)
        assert not fails
        # inference_data 2
        test_dict = {
            "posterior_predictive": ["y_hat"],
            "observed_data": ["y"],
            "sample_stats_prior": ["lp"],
            "sample_stats": ["lp"],
            "prior_predictive": ["y_hat"],
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
