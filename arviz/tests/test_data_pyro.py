# pylint: disable=no-member, invalid-name, redefined-outer-name
import numpy as np
import packaging
import pytest
import torch
import pyro
from pyro.infer import Predictive

from ..data.io_pyro import from_pyro
from .helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    load_cached_models,
)


class TestDataPyro:
    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        class Data:
            obj = load_cached_models(eight_schools_params, draws, chains, "pyro")["pyro"]

        return Data

    def get_inference_data(self, data, eight_schools_params):
        posterior_samples = data.obj.get_samples()
        model = data.obj.kernel.model
        posterior_predictive = Predictive(model, posterior_samples).get_samples(
            eight_schools_params["J"], torch.from_numpy(eight_schools_params["sigma"]).float()
        )
        prior = Predictive(model, num_samples=500).get_samples(
            eight_schools_params["J"], torch.from_numpy(eight_schools_params["sigma"]).float()
        )
        return from_pyro(
            posterior=data.obj,
            prior=prior,
            posterior_predictive=posterior_predictive,
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
        )

    def test_inference_data(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        test_dict = {
            "posterior": ["mu", "tau", "eta"],
            "sample_stats": ["diverging"],
            "posterior_predictive": ["obs"],
            "prior": ["mu", "tau", "eta"],
            "prior_predictive": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    @pytest.mark.skipif(
        packaging.version.parse(pyro.__version__) < packaging.version.parse("1.0.0"),
        reason="requires pyro 1.0.0 or higher",
    )
    def test_inference_data_has_log_likelihood_and_observed_data(self, data):
        idata = from_pyro(data.obj)
        test_dict = {"sample_stats": ["log_likelihood"], "observed_data": ["obs"]}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

    def test_inference_data_no_posterior(self, data, eight_schools_params):
        posterior_samples = data.obj.get_samples()
        model = data.obj.kernel.model
        posterior_predictive = Predictive(model, posterior_samples).get_samples(
            eight_schools_params["J"], torch.from_numpy(eight_schools_params["sigma"]).float()
        )
        prior = Predictive(model, num_samples=500).get_samples(
            eight_schools_params["J"], torch.from_numpy(eight_schools_params["sigma"]).float()
        )
        idata = from_pyro(
            prior=prior,
            posterior_predictive=posterior_predictive,
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
        )
        test_dict = {"posterior_predictive": ["obs"], "prior": ["mu", "tau", "eta", "obs"]}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

    def test_inference_data_only_posterior(self, data):
        idata = from_pyro(data.obj)
        test_dict = {"posterior": ["mu", "tau", "eta"], "sample_stats": ["diverging"]}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

    @pytest.mark.skipif(
        packaging.version.parse(pyro.__version__) < packaging.version.parse("1.0.0"),
        reason="requires pyro 1.0.0 or higher",
    )
    def test_inference_data_only_posterior_has_log_likelihood(self, data):
        idata = from_pyro(data.obj)
        test_dict = {"sample_stats": ["log_likelihood"]}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails
