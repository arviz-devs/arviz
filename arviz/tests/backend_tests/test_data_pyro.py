# pylint: disable=no-member, invalid-name, redefined-outer-name
import numpy as np
import packaging
import pytest
import torch
import pyro
from pyro.infer import Predictive

from ...data.io_pyro import from_pyro
from ..helpers import (  # pylint: disable=unused-import
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
        posterior_predictive = Predictive(model, posterior_samples)(
            eight_schools_params["J"], torch.from_numpy(eight_schools_params["sigma"]).float()
        )
        prior = Predictive(model, num_samples=500)(
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
        test_dict = {"log_likelihood": ["obs"], "observed_data": ["obs"]}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

    def test_inference_data_no_posterior(self, data, eight_schools_params):
        posterior_samples = data.obj.get_samples()
        model = data.obj.kernel.model
        posterior_predictive = Predictive(model, posterior_samples)(
            eight_schools_params["J"], torch.from_numpy(eight_schools_params["sigma"]).float()
        )
        prior = Predictive(model, num_samples=500)(
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
        test_dict = {"log_likelihood": ["obs"]}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails

    def test_multiple_observed_rv(self):
        import pyro.distributions as dist
        from pyro.infer import MCMC, NUTS

        y1 = torch.randn(10)
        y2 = torch.randn(10)

        def model_example_multiple_obs(y1=None, y2=None):
            x = pyro.sample("x", dist.Normal(1, 3))
            pyro.sample("y1", dist.Normal(x, 1), obs=y1)
            pyro.sample("y2", dist.Normal(x, 1), obs=y2)

        nuts_kernel = NUTS(model_example_multiple_obs)
        mcmc = MCMC(nuts_kernel, num_samples=10)
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
