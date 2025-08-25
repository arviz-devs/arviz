# pylint: disable=no-member, invalid-name, redefined-outer-name, too-many-public-methods
from collections import namedtuple
import numpy as np
import pytest

from ...data.io_numpyro import from_numpyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    importorskip,
    load_cached_models,
)

# Skip all tests if jax or numpyro not installed
jax = importorskip("jax")
PRNGKey = jax.random.PRNGKey
numpyro = importorskip("numpyro")
Predictive = numpyro.infer.Predictive


class TestDataNumPyro:
    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        class Data:
            obj = load_cached_models(eight_schools_params, draws, chains, "numpyro")["numpyro"]

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
        model = data.obj.sampler.model
        predictions = Predictive(model, posterior_samples)(
            PRNGKey(2), predictions_params["J"], predictions_params["sigma"]
        )
        return predictions

    def get_inference_data(
        self, data, eight_schools_params, predictions_data, predictions_params, infer_dims=False
    ):
        posterior_samples = data.obj.get_samples()
        model = data.obj.sampler.model
        posterior_predictive = Predictive(model, posterior_samples)(
            PRNGKey(1), eight_schools_params["J"], eight_schools_params["sigma"]
        )
        prior = Predictive(model, num_samples=500)(
            PRNGKey(2), eight_schools_params["J"], eight_schools_params["sigma"]
        )
        dims = {"theta": ["school"], "eta": ["school"], "obs": ["school"]}
        pred_dims = {"theta": ["school_pred"], "eta": ["school_pred"], "obs": ["school_pred"]}
        if infer_dims:
            dims = None
            pred_dims = None

        predictions = predictions_data
        return from_numpyro(
            posterior=data.obj,
            prior=prior,
            posterior_predictive=posterior_predictive,
            predictions=predictions,
            coords={
                "school": np.arange(eight_schools_params["J"]),
                "school_pred": np.arange(predictions_params["J"]),
            },
            dims=dims,
            pred_dims=pred_dims,
        )

    def test_inference_data_namedtuple(self, data):
        samples = data.obj.get_samples()
        Samples = namedtuple("Samples", samples)
        data_namedtuple = Samples(**samples)
        _old_fn = data.obj.get_samples
        data.obj.get_samples = lambda *args, **kwargs: data_namedtuple
        inference_data = from_numpyro(
            posterior=data.obj,
            dims={},  # This mock test needs to turn off autodims like so or mock group_by_chain
        )
        assert isinstance(data.obj.get_samples(), Samples)
        data.obj.get_samples = _old_fn
        for key in samples:
            assert key in inference_data.posterior

    def test_inference_data(self, data, eight_schools_params, predictions_data, predictions_params):
        inference_data = self.get_inference_data(
            data, eight_schools_params, predictions_data, predictions_params
        )
        test_dict = {
            "posterior": ["mu", "tau", "eta"],
            "sample_stats": ["diverging"],
            "log_likelihood": ["obs"],
            "posterior_predictive": ["obs"],
            "predictions": ["obs"],
            "prior": ["mu", "tau", "eta"],
            "prior_predictive": ["obs"],
            "observed_data": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

        # test dims
        dims = inference_data.posterior_predictive.sizes["school"]
        pred_dims = inference_data.predictions.sizes["school_pred"]
        assert dims == 8
        assert pred_dims == 8

    def test_inference_data_no_posterior(
        self, data, eight_schools_params, predictions_data, predictions_params
    ):
        posterior_samples = data.obj.get_samples()
        model = data.obj.sampler.model
        posterior_predictive = Predictive(model, posterior_samples)(
            PRNGKey(1), eight_schools_params["J"], eight_schools_params["sigma"]
        )
        prior = Predictive(model, num_samples=500)(
            PRNGKey(2), eight_schools_params["J"], eight_schools_params["sigma"]
        )
        predictions = predictions_data
        constant_data = {"J": 8, "sigma": eight_schools_params["sigma"]}
        predictions_constant_data = predictions_params
        # only prior
        inference_data = from_numpyro(prior=prior)
        test_dict = {"prior": ["mu", "tau", "eta"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only prior: {fails}"
        # only posterior_predictive
        inference_data = from_numpyro(posterior_predictive=posterior_predictive)
        test_dict = {"posterior_predictive": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only posterior_predictive: {fails}"
        # only predictions
        inference_data = from_numpyro(predictions=predictions)
        test_dict = {"predictions": ["obs"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only predictions: {fails}"
        # only constant_data
        inference_data = from_numpyro(constant_data=constant_data)
        test_dict = {"constant_data": ["J", "sigma"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only constant_data: {fails}"
        # only predictions_constant_data
        inference_data = from_numpyro(predictions_constant_data=predictions_constant_data)
        test_dict = {"predictions_constant_data": ["J", "sigma"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails, f"only predictions_constant_data: {fails}"
        # prior and posterior_predictive
        idata = from_numpyro(
            prior=prior,
            posterior_predictive=posterior_predictive,
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
        )
        test_dict = {"posterior_predictive": ["obs"], "prior": ["mu", "tau", "eta", "obs"]}
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails, f"prior and posterior_predictive: {fails}"

    def test_inference_data_only_posterior(self, data):
        idata = from_numpyro(data.obj)
        test_dict = {
            "posterior": ["mu", "tau", "eta"],
            "sample_stats": ["diverging"],
            "log_likelihood": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, idata)
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

    def test_inference_data_constant_data(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        x1 = 10
        x2 = 12
        y1 = np.random.randn(10)

        def model_constant_data(x, y1=None):
            _x = numpyro.sample("x", dist.Normal(1, 3))
            numpyro.sample("y1", dist.Normal(x * _x, 1), obs=y1)

        nuts_kernel = NUTS(model_constant_data)
        mcmc = MCMC(nuts_kernel, num_samples=10, num_warmup=2)
        mcmc.run(PRNGKey(0), x=x1, y1=y1)
        posterior = mcmc.get_samples()
        posterior_predictive = Predictive(model_constant_data, posterior)(PRNGKey(1), x1)
        predictions = Predictive(model_constant_data, posterior)(PRNGKey(2), x2)
        inference_data = from_numpyro(
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
        inference_data = from_numpyro(predictions=predictions, num_chains=chains)
        nchains = inference_data.predictions.sizes["chain"]
        assert nchains == chains

    @pytest.mark.parametrize("nchains", [1, 2])
    @pytest.mark.parametrize("thin", [1, 2, 3, 5, 10])
    def test_mcmc_with_thinning(self, nchains, thin):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        x = np.random.normal(10, 3, size=100)

        def model(x):
            numpyro.sample(
                "x",
                dist.Normal(
                    numpyro.sample("loc", dist.Uniform(0, 20)),
                    numpyro.sample("scale", dist.Uniform(0, 20)),
                ),
                obs=x,
            )

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=400, num_chains=nchains, thinning=thin)
        mcmc.run(PRNGKey(0), x=x)

        inference_data = from_numpyro(mcmc)
        assert inference_data.posterior["loc"].shape == (nchains, 400 // thin)

    def test_mcmc_improper_uniform(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        def model():
            x = numpyro.sample("x", dist.ImproperUniform(dist.constraints.positive, (), ()))
            return numpyro.sample("y", dist.Normal(x, 1), obs=1.0)

        mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
        mcmc.run(PRNGKey(0))
        inference_data = from_numpyro(mcmc)
        assert inference_data.observed_data

    def test_mcmc_infer_dims(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        def model():
            # note: group2 gets assigned dim=-1 and group1 is assigned dim=-2
            with numpyro.plate("group2", 5), numpyro.plate("group1", 10):
                _ = numpyro.sample("param", dist.Normal(0, 1))

        mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
        mcmc.run(PRNGKey(0))
        inference_data = from_numpyro(
            mcmc, coords={"group1": np.arange(10), "group2": np.arange(5)}
        )
        assert inference_data.posterior.param.dims == ("chain", "draw", "group1", "group2")
        assert all(dim in inference_data.posterior.param.coords for dim in ("group1", "group2"))

    def test_mcmc_infer_unsorted_dims(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        def model():
            group1_plate = numpyro.plate("group1", 10, dim=-1)
            group2_plate = numpyro.plate("group2", 5, dim=-2)

            # the plate contexts are entered in a different order than the pre-defined dims
            # we should make sure this still works because the trace has all of the info it needs
            with group2_plate, group1_plate:
                _ = numpyro.sample("param", dist.Normal(0, 1))

        mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
        mcmc.run(PRNGKey(0))
        inference_data = from_numpyro(
            mcmc, coords={"group1": np.arange(10), "group2": np.arange(5)}
        )
        assert inference_data.posterior.param.dims == ("chain", "draw", "group2", "group1")
        assert all(dim in inference_data.posterior.param.coords for dim in ("group1", "group2"))

    def test_mcmc_infer_dims_no_coords(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        def model():
            with numpyro.plate("group", 5):
                _ = numpyro.sample("param", dist.Normal(0, 1))

        mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
        mcmc.run(PRNGKey(0))
        inference_data = from_numpyro(mcmc)
        assert inference_data.posterior.param.dims == ("chain", "draw", "group")

    def test_mcmc_event_dims(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        def model():
            _ = numpyro.sample(
                "gamma", dist.ZeroSumNormal(1, event_shape=(10,)), infer={"event_dims": ["groups"]}
            )

        mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
        mcmc.run(PRNGKey(0))
        inference_data = from_numpyro(mcmc, coords={"groups": np.arange(10)})
        assert inference_data.posterior.gamma.dims == ("chain", "draw", "groups")
        assert "groups" in inference_data.posterior.gamma.coords

    @pytest.mark.xfail
    def test_mcmc_inferred_dims_univariate(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        import jax.numpy as jnp

        def model():
            alpha = numpyro.sample("alpha", dist.Normal(0, 1))
            sigma = numpyro.sample("sigma", dist.HalfNormal(1))
            with numpyro.plate("obs_idx", 3):
                # mu is plated by obs_idx, but isnt broadcasted to the plate shape
                # the expected behavior is that this should cause a failure
                mu = numpyro.deterministic("mu", alpha)
                return numpyro.sample("y", dist.Normal(mu, sigma), obs=jnp.array([-1, 0, 1]))

        mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
        mcmc.run(PRNGKey(0))
        inference_data = from_numpyro(mcmc, coords={"obs_idx": np.arange(3)})
        assert inference_data.posterior.mu.dims == ("chain", "draw", "obs_idx")
        assert "obs_idx" in inference_data.posterior.mu.coords

    def test_mcmc_extra_event_dims(self):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        def model():
            gamma = numpyro.sample("gamma", dist.ZeroSumNormal(1, event_shape=(10,)))
            _ = numpyro.deterministic("gamma_plus1", gamma + 1)

        mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=10)
        mcmc.run(PRNGKey(0))
        inference_data = from_numpyro(
            mcmc, coords={"groups": np.arange(10)}, extra_event_dims={"gamma_plus1": ["groups"]}
        )
        assert inference_data.posterior.gamma_plus1.dims == ("chain", "draw", "groups")
        assert "groups" in inference_data.posterior.gamma_plus1.coords

    def test_mcmc_predictions_infer_dims(
        self, data, eight_schools_params, predictions_data, predictions_params
    ):
        inference_data = self.get_inference_data(
            data, eight_schools_params, predictions_data, predictions_params, infer_dims=True
        )
        assert inference_data.predictions.obs.dims == ("chain", "draw", "J")
        assert "J" in inference_data.predictions.obs.coords

    def test_potential_energy_sign_conversion(self):
        """Test that potential energy is converted to log probability (lp) with correct sign."""
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        num_samples = 10

        def simple_model():
            numpyro.sample("x", dist.Normal(0, 1))

        nuts_kernel = NUTS(simple_model)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=5)
        mcmc.run(PRNGKey(0), extra_fields=["potential_energy"])

        # Get the raw extra fields from NumPyro
        extra_fields = mcmc.get_extra_fields(group_by_chain=True)
        # Convert to ArviZ InferenceData
        inference_data = from_numpyro(mcmc)
        arviz_lp = inference_data["sample_stats"]["lp"].values

        np.testing.assert_array_equal(arviz_lp, -extra_fields["potential_energy"])
