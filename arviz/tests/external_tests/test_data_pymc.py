# pylint: disable=no-member, invalid-name, redefined-outer-name, protected-access, too-many-public-methods
from sys import version_info
from typing import Dict, Tuple

import numpy as np
import pkg_resources
import packaging
import pandas as pd
import pytest
from numpy import ma

from ... import (  # pylint: disable=wrong-import-position
    InferenceData,
    from_dict,
    from_pymc3,
    from_pymc3_predictions,
)

from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    importorskip,
    load_cached_models,
)

# Skip all tests unless running on pymc3 v3
try:
    pymc3_version = pkg_resources.get_distribution("pymc3").version
    PYMC3_V4 = packaging.version.parse(pymc3_version) >= packaging.version.parse("4.0")
    PYMC3_installed = True
    import pymc3 as pm
except pkg_resources.DistributionNotFound:
    PYMC3_V4 = False
    PYMC3_installed = False

pytestmark = pytest.mark.skipif(
    not PYMC3_installed or PYMC3_V4,
    reason="Run tests only if pymc3 installed and its version is <4.0",
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
        """Test that we can add predictions to a previously-existing InferenceData."""
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

    def test_from_pymc_trace_inference_data(self):
        """Check if the error is raised successfully after passing InferenceData as trace"""
        idata = from_dict(
            posterior={"A": np.random.randn(2, 10, 2), "B": np.random.randn(2, 10, 5, 2)}
        )
        assert isinstance(idata, InferenceData)
        with pytest.raises(ValueError):
            from_pymc3(trace=idata, model=pm.Model())

    def test_from_pymc_predictions_new(self, data, eight_schools_params):
        # check creating new
        inference_data, posterior_predictive = self.make_predictions_inference_data(
            data, eight_schools_params
        )
        test_dict = {
            "posterior": ["mu", "tau", "eta", "theta"],
            "predictions": ["obs"],
            "~observed_data": [""],
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
                raise ValueError(f"Unexpected values shape for variable {key}")
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

    @pytest.mark.skipif(
        packaging.version.Version(pm.__version__) < packaging.version.Version("3.9.0"),
        reason="Requires PyMC3 >= 3.9.0",
    )
    @pytest.mark.parametrize("use_context", [True, False])
    def test_autodetect_coords_from_model(self, use_context):
        df_data = pd.DataFrame(columns=["date"]).set_index("date")
        dates = pd.date_range(start="2020-05-01", end="2020-05-20")
        for city, mu in {"Berlin": 15, "San Marino": 18, "Paris": 16}.items():
            df_data[city] = np.random.normal(loc=mu, size=len(dates))
        df_data.index = dates
        df_data.index.name = "date"

        coords = {"date": df_data.index, "city": df_data.columns}
        with pm.Model(coords=coords) as model:
            europe_mean = pm.Normal("europe_mean_temp", mu=15.0, sd=3.0)
            city_offset = pm.Normal("city_offset", mu=0.0, sd=3.0, dims="city")
            city_temperature = pm.Deterministic(
                "city_temperature", europe_mean + city_offset, dims="city"
            )

            data_dims = ("date", "city")
            data = pm.Data("data", df_data, dims=data_dims)
            _ = pm.Normal("likelihood", mu=city_temperature, sd=0.5, observed=data, dims=data_dims)

            trace = pm.sample(
                return_inferencedata=False,
                compute_convergence_checks=False,
                cores=1,
                chains=1,
                tune=20,
                draws=30,
                step=pm.Metropolis(),
            )
            if use_context:
                idata = from_pymc3(trace=trace)
        if not use_context:
            idata = from_pymc3(trace=trace, model=model)

        assert "city" in list(idata.posterior.dims)
        assert "city" in list(idata.observed_data.dims)
        assert "date" in list(idata.observed_data.dims)
        np.testing.assert_array_equal(idata.posterior.coords["city"], coords["city"])
        np.testing.assert_array_equal(idata.observed_data.coords["date"], coords["date"])
        np.testing.assert_array_equal(idata.observed_data.coords["city"], coords["city"])

    def test_ovewrite_model_coords_dims(self):
        """Check coords and dims from model object can be partially overwrited."""
        dim1 = ["a", "b"]
        new_dim1 = ["c", "d"]
        coords = {"dim1": dim1, "dim2": ["c1", "c2"]}
        x_data = np.arange(4).reshape((2, 2))
        y = x_data + np.random.normal(size=(2, 2))
        with pm.Model(coords=coords):
            x = pm.Data("x", x_data, dims=("dim1", "dim2"))
            beta = pm.Normal("beta", 0, 1, dims="dim1")
            _ = pm.Normal("obs", x * beta, 1, observed=y, dims=("dim1", "dim2"))
            trace = pm.sample(100, tune=100)
            idata1 = from_pymc3(trace)
            idata2 = from_pymc3(trace, coords={"dim1": new_dim1}, dims={"beta": ["dim2"]})

        test_dict = {"posterior": ["beta"], "observed_data": ["obs"], "constant_data": ["x"]}
        fails1 = check_multiple_attrs(test_dict, idata1)
        assert not fails1
        fails2 = check_multiple_attrs(test_dict, idata2)
        assert not fails2
        assert "dim1" in list(idata1.posterior.beta.dims)
        assert "dim2" in list(idata2.posterior.beta.dims)
        assert np.all(idata1.constant_data.x.dim1.values == np.array(dim1))
        assert np.all(idata1.constant_data.x.dim2.values == np.array(["c1", "c2"]))
        assert np.all(idata2.constant_data.x.dim1.values == np.array(new_dim1))
        assert np.all(idata2.constant_data.x.dim2.values == np.array(["c1", "c2"]))

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

    def test_mv_missing_data_model(self):
        data = ma.masked_values([[1, 2], [2, 2], [-1, 4], [2, -1], [-1, -1]], value=-1)

        model = pm.Model()
        with model:
            mu = pm.Normal("mu", 0, 1, shape=2)
            sd_dist = pm.HalfNormal.dist(1.0)
            chol, *_ = pm.LKJCholeskyCov("chol_cov", n=2, eta=1, sd_dist=sd_dist, compute_corr=True)
            pm.MvNormal("y", mu=mu, chol=chol, observed=data)
            trace = pm.sample(100, chains=2)

        # make sure that data is really missing
        (y_missing,) = model.missing_values
        assert y_missing.tag.test_value.shape == (4,)
        inference_data = from_pymc3(trace=trace, model=model)
        test_dict = {
            "posterior": ["mu", "chol_cov"],
            "observed_data": ["y"],
            "log_likelihood": ["y"],
        }
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
            "sample_stats": ["diverging", "lp", "~log_likelihood"],
        }
        if not log_likelihood:
            test_dict.pop("log_likelihood")
            test_dict["~log_likelihood"] = [""]
        if isinstance(log_likelihood, list):
            test_dict["log_likelihood"] = ["y1", "~y2"]

        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

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
        test_dict = {
            "posterior": ["mu"],
            "sample_stats": ["lp"],
            "log_likelihood": ["x"],
            "observed_data": ["value", "~x"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        assert inference_data.observed_data.value.dtype.kind == "f"

    @pytest.mark.parametrize("multiobs", (True, False))
    def test_multiobservedrv_to_observed_data(self, multiobs):
        # fake regression data, with weights (W)
        np.random.seed(2019)
        N = 100
        X = np.random.uniform(size=N)
        W = 1 + np.random.poisson(size=N)
        a, b = 5, 17
        Y = a + np.random.normal(b * X)

        with pm.Model():
            a = pm.Normal("a", 0, 10)
            b = pm.Normal("b", 0, 10)
            mu = a + b * X
            sigma = pm.HalfNormal("sigma", 1)

            def weighted_normal(y, w):
                return w * pm.Normal.dist(mu=mu, sd=sigma).logp(y)

            y_logp = pm.DensityDist(  # pylint: disable=unused-variable
                "y_logp", weighted_normal, observed={"y": Y, "w": W}
            )
            trace = pm.sample(20, tune=20)
            idata = from_pymc3(trace, density_dist_obs=multiobs)
        multiobs_str = "" if multiobs else "~"
        test_dict = {
            "posterior": ["a", "b", "sigma"],
            "sample_stats": ["lp"],
            "log_likelihood": ["y_logp"],
            f"{multiobs_str}observed_data": ["y", "w"],
        }
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails
        if multiobs:
            assert idata.observed_data.y.dtype.kind == "f"

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
        test_dict = {"posterior": ["beta"], "~observed_data": [""]}
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
            with model:  # pylint: disable=not-context-manager
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

        with pytest.warns(FutureWarning, match="without the model"):
            inference_data = from_pymc3(prior=prior)
        test_dict = {
            "prior": ["beta", "obs"],
            "~prior_predictive": [""],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_multivariate_observations(self):
        coords = {"direction": ["x", "y", "z"], "experiment": np.arange(20)}
        data = np.random.multinomial(20, [0.2, 0.3, 0.5], size=20)
        with pm.Model(coords=coords):
            p = pm.Beta("p", 1, 1, shape=(3,))
            pm.Multinomial("y", 20, p, dims=("experiment", "direction"), observed=data)
            idata = pm.sample(draws=50, tune=100, return_inferencedata=True)
        test_dict = {
            "posterior": ["p"],
            "sample_stats": ["lp"],
            "log_likelihood": ["y"],
            "observed_data": ["y"],
        }
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails
        assert "direction" not in idata.log_likelihood.dims
        assert "direction" in idata.observed_data.dims


class TestPyMC3WarmupHandling:
    @pytest.mark.skipif(
        not hasattr(pm.backends.base.SamplerReport, "n_draws"),
        reason="requires pymc3 3.9 or higher",
    )
    @pytest.mark.parametrize("save_warmup", [False, True])
    @pytest.mark.parametrize("chains", [1, 2])
    @pytest.mark.parametrize("tune,draws", [(0, 50), (10, 40), (30, 0)])
    def test_save_warmup(self, save_warmup, chains, tune, draws):
        with pm.Model():
            pm.Uniform("u1")
            pm.Normal("n1")
            trace = pm.sample(
                tune=tune,
                draws=draws,
                chains=chains,
                cores=1,
                step=pm.Metropolis(),
                discard_tuned_samples=False,
            )
            assert isinstance(trace, pm.backends.base.MultiTrace)
            idata = from_pymc3(trace, save_warmup=save_warmup)
        warmup_prefix = "" if save_warmup and (tune > 0) else "~"
        post_prefix = "" if draws > 0 else "~"
        test_dict = {
            f"{post_prefix}posterior": ["u1", "n1"],
            f"{post_prefix}sample_stats": ["~tune", "accept"],
            f"{warmup_prefix}warmup_posterior": ["u1", "n1"],
            f"{warmup_prefix}warmup_sample_stats": ["~tune"],
            "~warmup_log_likelihood": [""],
            "~log_likelihood": [""],
        }
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails
        if hasattr(idata, "posterior"):
            assert idata.posterior.dims["chain"] == chains
            assert idata.posterior.dims["draw"] == draws
        if hasattr(idata, "warmup_posterior"):
            assert idata.warmup_posterior.dims["chain"] == chains
            assert idata.warmup_posterior.dims["draw"] == tune

    @pytest.mark.skipif(
        hasattr(pm.backends.base.SamplerReport, "n_draws"),
        reason="requires pymc3 3.8 or lower",
    )
    def test_save_warmup_issue_1208_before_3_9(self):
        with pm.Model():
            pm.Uniform("u1")
            pm.Normal("n1")
            trace = pm.sample(
                tune=100,
                draws=200,
                chains=2,
                cores=1,
                step=pm.Metropolis(),
                discard_tuned_samples=False,
            )
            assert isinstance(trace, pm.backends.base.MultiTrace)
            assert len(trace) == 300

            # <=3.8 did not track n_draws in the sampler report,
            # making from_pymc3 fall back to len(trace) and triggering a warning
            with pytest.warns(UserWarning, match="Warmup samples"):
                idata = from_pymc3(trace, save_warmup=True)
        test_dict = {
            "posterior": ["u1", "n1"],
            "sample_stats": ["~tune", "accept"],
            "~warmup_posterior": [""],
            "~warmup_sample_stats": [""],
        }
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails
        assert idata.posterior.dims["draw"] == 300
        assert idata.posterior.dims["chain"] == 2

    @pytest.mark.skipif(
        not hasattr(pm.backends.base.SamplerReport, "n_draws"),
        reason="requires pymc3 3.9 or higher",
    )
    def test_save_warmup_issue_1208_after_3_9(self):
        with pm.Model():
            pm.Uniform("u1")
            pm.Normal("n1")
            trace = pm.sample(
                tune=100,
                draws=200,
                chains=2,
                cores=1,
                step=pm.Metropolis(),
                discard_tuned_samples=False,
            )
            assert isinstance(trace, pm.backends.base.MultiTrace)
            assert len(trace) == 300

            # from original trace, warmup draws should be separated out
            idata = from_pymc3(trace, save_warmup=True)
            test_dict = {
                "posterior": ["u1", "n1"],
                "sample_stats": ["~tune", "accept"],
                "warmup_posterior": ["u1", "n1"],
                "warmup_sample_stats": ["~tune", "accept"],
            }
            fails = check_multiple_attrs(test_dict, idata)
            assert not fails
            assert idata.posterior.dims["chain"] == 2
            assert idata.posterior.dims["draw"] == 200

            # manually sliced trace triggers the same warning as <=3.8
            with pytest.warns(UserWarning, match="Warmup samples"):
                idata = from_pymc3(trace[-30:], save_warmup=True)
            test_dict = {
                "posterior": ["u1", "n1"],
                "sample_stats": ["~tune", "accept"],
                "~warmup_posterior": [""],
                "~warmup_sample_stats": [""],
            }
            fails = check_multiple_attrs(test_dict, idata)
            assert not fails
            assert idata.posterior.dims["chain"] == 2
            assert idata.posterior.dims["draw"] == 30
