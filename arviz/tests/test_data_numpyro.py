# pylint: disable=no-member, invalid-name, redefined-outer-name
import numpy as np
import pytest
from jax.random import PRNGKey
from numpyro.infer import Predictive

from ..data.io_numpyro import from_numpyro
from .helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    load_cached_models,
)


class TestDataNumPyro:
    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        class Data:
            obj = load_cached_models(eight_schools_params, draws, chains, "numpyro")["numpyro"]

        return Data

    def get_inference_data(self, data, eight_schools_params):
        posterior_samples = data.obj.get_samples()
        model = data.obj.sampler.model
        posterior_predictive = Predictive(model, posterior_samples).get_samples(
            PRNGKey(1), eight_schools_params["J"], eight_schools_params["sigma"]
        )
        prior = Predictive(model, num_samples=500).get_samples(
            PRNGKey(2), eight_schools_params["J"], eight_schools_params["sigma"]
        )
        return from_numpyro(
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
            "sample_stats": ["diverging", "tree_size", "depth", "log_likelihood"],
            "posterior_predictive": ["obs"],
            "prior": ["mu", "tau", "eta"],
            "prior_predictive": ["obs"],
            "observed_data": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
