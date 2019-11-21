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

        return (
            from_pymc3(
                trace=data.obj,
                prior=prior,
                posterior_predictive=posterior_predictive,
                coords={"school": np.arange(eight_schools_params["J"])},
                dims={"theta": ["school"], "eta": ["school"]},
            ),
            posterior_predictive,
        )

    def test_from_pymc(self, data, eight_schools_params, chains, draws):
        inference_data, posterior_predictive = self.get_inference_data(data, eight_schools_params)
        test_dict = {
            "posterior": ["mu", "tau", "eta", "theta"],
            "sample_stats": ["diverging", "log_likelihood"],
            "posterior_predictive": ["obs"],
            "prior": ["mu", "tau", "eta", "theta"],
            "prior_predictive": ["obs"],
            "observed_data": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        for key, values in posterior_predictive.items():
            ivalues = inference_data.posterior_predictive[key]
            for chain in range(chains):
                assert np.all(
                    np.isclose(ivalues[chain], values[chain * draws : (chain + 1) * draws])
                )

    def test_posterior_predictive_keep_size(self, data, chains, draws, eight_schools_params):
        with data.model:
            posterior_predictive = pm.sample_posterior_predictive(data.obj, keep_size=True)

        inference_data = from_pymc3(
            trace=data.obj,
            posterior_predictive=posterior_predictive,
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
        )
        shape = inference_data.posterior_predictive.obs.shape
        assert np.all(
            [obs_s == s for obs_s, s in zip(shape, (chains, draws, eight_schools_params["J"]))]
        )

    def test_posterior_predictive_warning(self, data, eight_schools_params, caplog):
        with data.model:
            posterior_predictive = pm.sample_posterior_predictive(data.obj, 370)

        inference_data = from_pymc3(
            trace=data.obj,
            posterior_predictive=posterior_predictive,
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
        )
        records = caplog.records
        shape = inference_data.posterior_predictive.obs.shape
        assert np.all([obs_s == s for obs_s, s in zip(shape, (1, 370, eight_schools_params["J"]))])
        assert len(records) == 1
        assert records[0].levelname == "WARNING"

    def test_missing_data_model(self):
        # source pymc3/pymc3/tests/test_missing.py
        data = ma.masked_values([1, 2, -1, 4, -1], value=-1)
        model = pm.Model()
        with model:
            x = pm.Normal("x", 1, 1)
            pm.Normal("y", x, 1, observed=data)
            trace = pm.sample(100, chains=2)

        # make sure that data is really missing
        (y_missing,) = model.missing_values
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

    def test_multiple_observed_rv_without_observations(self):
        with pm.Model():
            mu = pm.Normal("mu")
            x = pm.DensityDist(  # pylint: disable=unused-variable
                "x", pm.Normal.dist(mu, 1.0).logp, observed={"value": 0.1}
            )
            trace = pm.sample(100, chains=2)
        inference_data = from_pymc3(trace=trace)
        assert inference_data
        assert not hasattr(inference_data, "observed_data")
        assert hasattr(inference_data, "posterior")
        assert hasattr(inference_data, "sample_stats")

    def test_single_observation(self):
        with pm.Model():
            p = pm.Uniform("p", 0, 1)
            pm.Binomial("w", p=p, n=2, observed=1)
            trace = pm.sample(500, chains=2)

        inference_data = from_pymc3(trace=trace)
        assert inference_data

    def test_constant_data(self):
        with pm.Model():
            x = pm.Data("x", [1.0, 2.0, 3.0])
            y = pm.Data("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 1)
            obs = pm.Normal("obs", x * beta, 1, observed=y)  # pylint: disable=unused-variable
            trace = pm.sample(100, tune=100)

        inference_data = from_pymc3(trace=trace)
        test_dict = {"posterior": ["beta"], "observed_data": ["obs"], "constant_data": ["x"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_no_trace(self):
        with pm.Model():
            x = pm.Data("x", [1.0, 2.0, 3.0])
            y = pm.Data("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 1)
            obs = pm.Normal("obs", x * beta, 1, observed=y)  # pylint: disable=unused-variable
            trace = pm.sample(100, tune=100)
            prior = pm.sample_prior_predictive()
            posterior_predictive = pm.sample_posterior_predictive(trace)

        # Only prior
        inference_data = from_pymc3(prior=prior)
        test_dict = {"prior": ["beta", "obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        # Only posterior_predictive
        inference_data = from_pymc3(posterior_predictive=posterior_predictive)
        test_dict = {"posterior_predictive": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        # Prior and posterior_predictive but no trace
        inference_data = from_pymc3(prior=prior, posterior_predictive=posterior_predictive)
        test_dict = {"prior": ["beta", "obs"], "posterior_predictive": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
