# pylint: disable=no-member, invalid-name, redefined-outer-name
import numpy as np
import packaging
import pytest

from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    importorskip,
    load_cached_models,
)

# Skip all tests if pyro or pytorch not installed
torch = importorskip("torch")
pyro = importorskip("pyro")
Predictive = pyro.infer.Predictive
dist = pyro.distributions


class TestDataPyro:
    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        class Data:
            obj = load_cached_models(eight_schools_params, draws, chains, "pyro")["pyro"]

        return Data

    @pytest.fixture(scope="class")
    def predictions_params(self):
        """Predictions data for eight schools."""
        return {
            "J": 8,
            "sigma": np.array([5.0, 7.0, 12.0, 4.0, 6.0, 10.0, 3.0, 9.0]),
        }

    @pytest.fixture(scope="class")
    def predictions_data(self, data, predictions_params):
        """Generate predictions for predictions_params"""
        posterior_samples = data.obj.get_samples()
        model = data.obj.kernel.model
        predictions = Predictive(model, posterior_samples)(
            predictions_params["J"], torch.from_numpy(predictions_params["sigma"]).float()
        )
        return predictions

    def get_inference_data(self, data, eight_schools_params, predictions_data):
        posterior_samples = data.obj.get_samples()
        model = data.obj.kernel.model
        posterior_predictive = Predictive(model, posterior_samples)(
            eight_schools_params["J"], torch.from_numpy(eight_schools_params["sigma"]).float()
        )
        prior = Predictive(model, num_samples=500)(
            eight_schools_params["J"], torch.from_numpy(eight_schools_params["sigma"]).float()
        )
        predictions = predictions_data
        return from_pyro(
            posterior=data.obj,
            prior=prior,
            posterior_predictive=posterior_predictive,
            predictions=predictions,
            coords={
                "school": np.arange(eight_schools_params["J"]),
                "school_pred": np.arange(eight_schools_params["J"]),
            },
            dims={"theta": ["school"], "eta": ["school"], "obs": ["school"]},
            pred_dims={"theta": ["school_pred"], "eta": ["school_pred"], "obs": ["school_pred"]},
        )

    def test_inference_data(self, data, eight_schools_params, predictions_data):
        inference_data = self.get_inference_data(data, eight_schools_params, predictions_data)
        test_dict = {
            "posterior": ["mu", "tau", "eta"],
            "sample_stats": ["diverging"],
            "posterior_predictive": ["obs"],
            "predictions": ["obs"],
            "prior": ["mu", "tau", "eta"],
            "prior_predictive": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

        # test dims
        dims = inference_data.posterior_predictive.sizes["school"]
        pred_dims = inference_data.predictions.sizes["school_pred"]
        assert dims == 8
        assert pred_dims == 8

    @pytest.mark.skipif(
        packaging.version.parse(pyro.__version__) < packaging.version.parse("1.0.0"),
        reason="requires pyro 1.0.0 or higher",
    )
    def test_inference_data_has_log_likelihood_and_observed_data(self, data):
        idata = from_pyro(data.obj)
        test_dict = {"log_likelihood": ["obs"], "observed_data": ["obs"]}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

    def test_inference_data_no_posterior(
        self, data, eight_schools_params, predictions_data, predictions_params
    ):
        posterior_samples = data.obj.get_samples()
        model = data.obj.kernel.model
        posterior_predictive = Predictive(model, posterior_samples)(
            eight_schools_params["J"], torch.from_numpy(eight_schools_params["sigma"]).float()
        )
        prior = Predictive(model, num_samples=500)(
            eight_schools_params["J"], torch.from_numpy(eight_schools_params["sigma"]).float()
        )
        predictions = predictions_data
        constant_data = {"J": 8, "sigma": eight_schools_params["sigma"]}
        predictions_constant_data = predictions_params
        # only prior
        inference_data = from_pyro(prior=prior)
        test_dict = {"prior": ["mu", "tau", "eta"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only prior: {fails}"
        # only posterior_predictive
        inference_data = from_pyro(posterior_predictive=posterior_predictive)
        test_dict = {"posterior_predictive": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only posterior_predictive: {fails}"
        # only predictions
        inference_data = from_pyro(predictions=predictions)
        test_dict = {"predictions": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only predictions: {fails}"
        # only constant_data
        inference_data = from_pyro(constant_data=constant_data)
        test_dict = {"constant_data": ["J", "sigma"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only constant_data: {fails}"
        # only predictions_constant_data
        inference_data = from_pyro(predictions_constant_data=predictions_constant_data)
        test_dict = {"predictions_constant_data": ["J", "sigma"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only predictions_constant_data: {fails}"
        # prior and posterior_predictive
        idata = from_pyro(
            prior=prior,
            posterior_predictive=posterior_predictive,
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
        )
        test_dict = {"posterior_predictive": ["obs"], "prior": ["mu", "tau", "eta", "obs"]}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails, f"prior and posterior_predictive: {fails}"

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
        test_dict = {"log_likelihood": ["obs"]}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

    def test_multiple_observed_rv(self):
        y1 = torch.randn(10)
        y2 = torch.randn(10)

        def model_example_multiple_obs(y1=None, y2=None):
            x = pyro.sample("x", dist.Normal(1, 3))
            pyro.sample("y1", dist.Normal(x, 1), obs=y1)
            pyro.sample("y2", dist.Normal(x, 1), obs=y2)

        nuts_kernel = pyro.infer.NUTS(model_example_multiple_obs)
        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=10)
        mcmc.run(y1=y1, y2=y2)
        inference_data = from_pyro(mcmc)
        test_dict = {
            "posterior": ["x"],
            "sample_stats": ["diverging"],
            "log_likelihood": ["y1", "y2"],
            "observed_data": ["y1", "y2"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        assert not hasattr(inference_data.sample_stats, "log_likelihood")

    def test_inference_data_constant_data(self):
        x1 = 10
        x2 = 12
        y1 = torch.randn(10)

        def model_constant_data(x, y1=None):
            _x = pyro.sample("x", dist.Normal(1, 3))
            pyro.sample("y1", dist.Normal(x * _x, 1), obs=y1)

        nuts_kernel = pyro.infer.NUTS(model_constant_data)
        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=10)
        mcmc.run(x=x1, y1=y1)
        posterior = mcmc.get_samples()
        posterior_predictive = Predictive(model_constant_data, posterior)(x1)
        predictions = Predictive(model_constant_data, posterior)(x2)
        inference_data = from_pyro(
            mcmc,
            posterior_predictive=posterior_predictive,
            predictions=predictions,
            constant_data={"x1": x1},
            predictions_constant_data={"x2": x2},
        )
        test_dict = {
            "posterior": ["x"],
            "posterior_predictive": ["y1"],
            "sample_stats": ["diverging"],
            "log_likelihood": ["y1"],
            "predictions": ["y1"],
            "observed_data": ["y1"],
            "constant_data": ["x1"],
            "predictions_constant_data": ["x2"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_inference_data_num_chains(self, predictions_data, chains):
        predictions = predictions_data
        inference_data = from_pyro(predictions=predictions, num_chains=chains)
        nchains = inference_data.predictions.sizes["chain"]
        assert nchains == chains

    @pytest.mark.parametrize("log_likelihood", [True, False])
    def test_log_likelihood(self, log_likelihood):
        """Test behaviour when log likelihood cannot be retrieved.

        If log_likelihood=True there is a warning to say log_likelihood group is skipped,
        if log_likelihood=False there is no warning and log_likelihood is skipped.
        """
        x = torch.randn((10, 2))
        y = torch.randn(10)

        def model_constant_data(x, y=None):
            beta = pyro.sample("beta", dist.Normal(torch.ones(2), 3))
            pyro.sample("y", dist.Normal(x.matmul(beta), 1), obs=y)

        nuts_kernel = pyro.infer.NUTS(model_constant_data)
        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=10)
        mcmc.run(x=x, y=y)
        if log_likelihood:
            with pytest.warns(UserWarning, match="Could not get vectorized trace"):
                inference_data = from_pyro(mcmc, log_likelihood=log_likelihood)
        else:
            inference_data = from_pyro(mcmc, log_likelihood=log_likelihood)
        test_dict = {
            "posterior": ["beta"],
            "sample_stats": ["diverging"],
            "~log_likelihood": [""],
            "observed_data": ["y"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
