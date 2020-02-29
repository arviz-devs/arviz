# pylint: disable=no-member, invalid-name, redefined-outer-name
import numpy as np
import pytest
from jax.random import PRNGKey
from numpyro.infer import Predictive

from ...data.io_numpyro import from_numpyro
from ..helpers import (  # pylint: disable=unused-import
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
        posterior_predictive = Predictive(model, posterior_samples)(
            PRNGKey(1), eight_schools_params["J"], eight_schools_params["sigma"]
        )
        prior = Predictive(model, num_samples=500)(
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
            "sample_stats": ["diverging", "tree_size", "depth"],
            "log_likelihood": ["obs"],
            "posterior_predictive": ["obs"],
            "prior": ["mu", "tau", "eta"],
            "prior_predictive": ["obs"],
            "observed_data": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_multiple_observed_rv(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        y1 = np.random.randn(10)
        y2 = np.random.randn(100)

        def model_example_multiple_obs(y1=None, y2=None):
            x = numpyro.sample("x", dist.Normal(1, 3))
            numpyro.sample("y1", dist.Normal(x, 1), obs=y1)
            numpyro.sample("y2", dist.Normal(x, 1), obs=y2)

        nuts_kernel = NUTS(model_example_multiple_obs)
        mcmc = MCMC(nuts_kernel, num_samples=10, num_warmup=2)
        mcmc.run(PRNGKey(0), y1=y1, y2=y2)
        inference_data = from_numpyro(mcmc)
        test_dict = {
            "posterior": ["x"],
            "sample_stats": ["diverging"],
            "log_likelihood": ["y1", "y2"],
            "observed_data": ["y1", "y2"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        #         from ..stats import waic
        #         waic_results = waic(inference_data)
        #         print(waic_results)
        #         print(waic_results.keys())
        #         print(waic_results.waic, waic_results.waic_se)
        assert not fails
        assert not hasattr(inference_data.sample_stats, "log_likelihood")
