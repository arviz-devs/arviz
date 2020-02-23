# pylint: disable=no-member, invalid-name, redefined-outer-name
from sys import version_info
from typing import Tuple, Dict
import pytest


import numpy as np
from numpy import ma
import pymc3 as pm

from arviz import from_pymc3, from_pymc3_predictions, InferenceData
from ..helpers import (  # pylint: disable=unused-import
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

    def get_predictions_inference_data(
        self, data, eight_schools_params, inplace
    ) -> Tuple[InferenceData, Dict[str, np.ndarray]]:
        with data.model:
            prior = pm.sample_prior_predictive()
            posterior_predictive = pm.sample_posterior_predictive(data.obj)

            idata = from_pymc3(
                trace=data.obj,
                prior=prior,
                coords={"school": np.arange(eight_schools_params["J"])},
                dims={"theta": ["school"], "eta": ["school"]},
            )
            assert isinstance(idata, InferenceData)
            extended = from_pymc3_predictions(
                posterior_predictive, idata_orig=idata, inplace=inplace
            )
            assert isinstance(extended, InferenceData)
            assert (id(idata) == id(extended)) == inplace
        return (extended, posterior_predictive)

    def make_predictions_inference_data(
        self, data, eight_schools_params
    ) -> Tuple[InferenceData, Dict[str, np.ndarray]]:
        with data.model:
            posterior_predictive = pm.sample_posterior_predictive(data.obj)
            idata = from_pymc3_predictions(
                posterior_predictive,
                posterior_trace=data.obj,
                coords={"school": np.arange(eight_schools_params["J"])},
                dims={"theta": ["school"], "eta": ["school"]},
            )
            assert isinstance(idata, InferenceData)
        return idata, posterior_predictive

    def test_from_pymc(self, data, eight_schools_params, chains, draws):
        inference_data, posterior_predictive = self.get_inference_data(data, eight_schools_params)
        test_dict = {
            "posterior": ["mu", "tau", "eta", "theta"],
            "sample_stats": ["diverging", "lp", "~log_likelihood"],
            "log_likelihood": ["obs"],
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

    def test_from_pymc_predictions(self, data, eight_schools_params):
        "Test that we can add predictions to a previously-existing InferenceData."
        test_dict = {
            "posterior": ["mu", "tau", "eta", "theta"],
            "sample_stats": ["diverging", "lp"],
            "log_likelihood": ["obs"],
            "predictions": ["obs"],
            "prior": ["mu", "tau", "eta", "theta"],
            "observed_data": ["obs"],
        }

        # check adding non-destructively
        inference_data, posterior_predictive = self.get_predictions_inference_data(
            data, eight_schools_params, False
        )
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        for key, values in posterior_predictive.items():
            ivalues = inference_data.predictions[key]
            assert ivalues.shape[0] == 1  # one chain in predictions
            assert np.all(np.isclose(ivalues[0], values))

        # check adding in place
        inference_data, posterior_predictive = self.get_predictions_inference_data(
            data, eight_schools_params, True
        )
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        for key, values in posterior_predictive.items():
            ivalues = inference_data.predictions[key]
            assert ivalues.shape[0] == 1  # one chain in predictions
            assert np.all(np.isclose(ivalues[0], values))

    def test_from_pymc_predictions_new(self, data, eight_schools_params):
        # check creating new
        inference_data, posterior_predictive = self.make_predictions_inference_data(
            data, eight_schools_params
        )
        test_dict = {
            "posterior": ["mu", "tau", "eta", "theta"],
            "predictions": ["obs"],
            "~observed_data": "",
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        for key, values in posterior_predictive.items():
            ivalues = inference_data.predictions[key]
            # could the following better be done by simply flattening both the ivalues
            # and the values?
            if len(ivalues.shape) == 3:
                ivalues_arr = np.reshape(
                    ivalues.values, (ivalues.shape[0] * ivalues.shape[1], ivalues.shape[2])
                )
            elif len(ivalues.shape) == 2:
                ivalues_arr = np.reshape(ivalues.values, (ivalues.shape[0] * ivalues.shape[1]))
            else:
                raise ValueError("Unexpected values shape for variable %s" % key)
            assert (ivalues.shape[0] == 2) and (ivalues.shape[1] == 500)
            assert values.shape[0] == 1000
            assert np.all(np.isclose(ivalues_arr, values))

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
        inference_data = from_pymc3(trace=trace, model=model)
        test_dict = {"posterior": ["x"], "observed_data": ["y"], "log_likelihood": ["y"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    @pytest.mark.parametrize("log_likelihood", [True, False, ["y1"]])
    def test_multiple_observed_rv(self, log_likelihood):
        y1_data = np.random.randn(10)
        y2_data = np.random.randn(100)
        with pm.Model():
            x = pm.Normal("x", 1, 1)
            pm.Normal("y1", x, 1, observed=y1_data)
            pm.Normal("y2", x, 1, observed=y2_data)
            trace = pm.sample(100, chains=2)
            inference_data = from_pymc3(trace=trace, log_likelihood=log_likelihood)
        test_dict = {
            "posterior": ["x"],
            "observed_data": ["y1", "y2"],
            "log_likelihood": ["y1", "y2"],
            "sample_stats": ["diverging", "lp"],
        }
        if not log_likelihood:
            test_dict.pop("log_likelihood")
            test_dict["~log_likelihood"] = []
        if isinstance(log_likelihood, list):
            test_dict["log_likelihood"] = ["y1", "~y2"]

        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        assert not hasattr(inference_data.sample_stats, "log_likelihood")

    @pytest.mark.skipif(
        version_info < (3, 6), reason="Requires updated PyMC3, which needs Python 3.6"
    )
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
        assert hasattr(inference_data, "log_likelihood")

    def test_single_observation(self):
        with pm.Model():
            p = pm.Uniform("p", 0, 1)
            pm.Binomial("w", p=p, n=2, observed=1)
            trace = pm.sample(500, chains=2)
            inference_data = from_pymc3(trace=trace)

        assert inference_data

    def test_potential(self):
        with pm.Model():
            x = pm.Normal("x", 0.0, 1.0)
            pm.Potential("z", pm.Normal.dist(x, 1.0).logp(np.random.randn(10)))
            trace = pm.sample(100, chains=2)
            inference_data = from_pymc3(trace=trace)

        assert inference_data

    @pytest.mark.parametrize("use_context", [True, False])
    def test_constant_data(self, use_context):
        """Test constant_data group behaviour."""
        with pm.Model():
            x = pm.Data("x", [1.0, 2.0, 3.0])
            y = pm.Data("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 1)
            obs = pm.Normal("obs", x * beta, 1, observed=y)  # pylint: disable=unused-variable
            trace = pm.sample(100, tune=100)
            if use_context:
                inference_data = from_pymc3(trace=trace)

        if not use_context:
            inference_data = from_pymc3(trace=trace)
        test_dict = {"posterior": ["beta"], "observed_data": ["obs"], "constant_data": ["x"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_predictions_constant_data(self):
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

        with pm.Model():
            x = pm.Data("x", [1.0, 2.0])
            y = pm.Data("y", [1.0, 2.0])
            beta = pm.Normal("beta", 0, 1)
            obs = pm.Normal("obs", x * beta, 1, observed=y)  # pylint: disable=unused-variable
            predictive_trace = pm.sample_posterior_predictive(trace)
            assert set(predictive_trace.keys()) == {"obs"}
            # this should be four chains of 100 samples
            # assert predictive_trace["obs"].shape == (400, 2)
            # but the shape seems to vary between pymc3 versions
            inference_data = from_pymc3_predictions(predictive_trace, posterior_trace=trace)
        test_dict = {"posterior": ["beta"], "~observed_data": ""}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, "Posterior data not copied over as expected."
        test_dict = {"predictions": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, "Predictions not instantiated as expected."
        test_dict = {"predictions_constant_data": ["x"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, "Predictions constant data not instantiated as expected."

    def test_no_trace(self):
        with pm.Model() as model:
            x = pm.Data("x", [1.0, 2.0, 3.0])
            y = pm.Data("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 1)
            obs = pm.Normal("obs", x * beta, 1, observed=y)  # pylint: disable=unused-variable
            trace = pm.sample(100, tune=100)
            prior = pm.sample_prior_predictive()
            posterior_predictive = pm.sample_posterior_predictive(trace)

        # Only prior
        inference_data = from_pymc3(prior=prior, model=model)
        test_dict = {"prior": ["beta"], "prior_predictive": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        # Only posterior_predictive
        inference_data = from_pymc3(posterior_predictive=posterior_predictive, model=model)
        test_dict = {"posterior_predictive": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        # Prior and posterior_predictive but no trace
        inference_data = from_pymc3(
            prior=prior, posterior_predictive=posterior_predictive, model=model
        )
        test_dict = {
            "prior": ["beta"],
            "prior_predictive": ["obs"],
            "posterior_predictive": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    @pytest.mark.parametrize("use_context", [True, False])
    def test_priors_with_model(self, use_context):
        """Test model is enough to get prior, prior predictive and observed_data."""
        with pm.Model() as model:
            x = pm.Data("x", [1.0, 2.0, 3.0])
            y = pm.Data("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 1)
            obs = pm.Normal("obs", x * beta, 1, observed=y)  # pylint: disable=unused-variable
            prior = pm.sample_prior_predictive()

        test_dict = {
            "prior": ["beta", "~obs"],
            "observed_data": ["obs"],
            "prior_predictive": ["obs"],
        }
        if use_context:
            with model:
                inference_data = from_pymc3(prior=prior)
        else:
            inference_data = from_pymc3(prior=prior, model=model)
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_no_model_deprecation(self):
        with pm.Model():
            x = pm.Data("x", [1.0, 2.0, 3.0])
            y = pm.Data("y", [1.0, 2.0, 3.0])
            beta = pm.Normal("beta", 0, 1)
            obs = pm.Normal("obs", x * beta, 1, observed=y)  # pylint: disable=unused-variable
            prior = pm.sample_prior_predictive()

        with pytest.warns(PendingDeprecationWarning, match="without the model"):
            inference_data = from_pymc3(prior=prior)
        test_dict = {
            "prior": ["beta", "obs"],
            "~prior_predictive": [],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
