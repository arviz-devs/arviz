# pylint: disable=no-member, invalid-name, redefined-outer-name
import numpy as np
from numpy import ma
import pymc3 as pm
import pytest

from arviz import from_pymc3
from .helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    load_cached_models,
)


class TestDataPyMC3:
    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        class Data:
            model, obj = load_cached_models(eight_schools_params, draws, chains, "pymc3")["pymc3"]

        return Data

    def get_inference_data(self, data, eight_schools_params):
        with data.model:
            prior = pm.sample_prior_predictive()
            posterior_predictive = pm.sample_posterior_predictive(data.obj)

        return from_pymc3(
            trace=data.obj,
            prior=prior,
            posterior_predictive=posterior_predictive,
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
        )

    def test_posterior(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        assert hasattr(inference_data, "posterior")

    def test_sampler_stats(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        assert hasattr(inference_data, "sample_stats")

    def test_posterior_predictive(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        assert hasattr(inference_data, "posterior_predictive")

    def test_prior(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        assert hasattr(inference_data, "prior")

    def test_missing_data_model(self):
        # source pymc3/pymc3/tests/test_missing.py
        data = ma.masked_values([1, 2, -1, 4, -1], value=-1)
        model = pm.Model()
        with model:
            x = pm.Normal("x", 1, 1)
            pm.Normal("y", x, 1, observed=data)
            trace = pm.sample(100, chains=2)

        # make sure that data is really missing
        y_missing, = model.missing_values
        assert y_missing.tag.test_value.shape == (2,)
        inference_data = from_pymc3(trace=trace)
        test_dict = {"posterior": ["x"], "observed_data": ["y"], "sample_stats": ["log_likelihood"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_multiple_observed_rv(self):
        y1_data = np.random.randn(10)
        y2_data = np.random.randn(100)
        with pm.Model():
            x = pm.Normal("x", 1, 1)
            pm.Normal("y1", x, 1, observed=y1_data)
            pm.Normal("y2", x, 1, observed=y2_data)
            trace = pm.sample(100, chains=2)
        inference_data = from_pymc3(trace=trace)
        test_dict = {"posterior": ["x"], "observed_data": ["y1", "y2"], "sample_stats": ["lp"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        assert not hasattr(inference_data.sample_stats, "log_likelihood")

    def test_single_observation(self):
        with pm.Model():
            p = pm.Uniform("p", 0, 1)
            pm.Binomial("w", p=p, n=2, observed=1)
            trace = pm.sample(500, chains=2)

        inference_data = from_pymc3(trace=trace)
        assert inference_data
