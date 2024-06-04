# pylint: disable=no-member, invalid-name, redefined-outer-name, too-many-function-args
import importlib
from collections import OrderedDict
import os

import numpy as np
import pytest

from ... import from_pystan

from ...data.io_pystan import get_draws, get_draws_stan3  # pylint: disable=unused-import
from ..helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    importorskip,
    load_cached_models,
    pystan_version,
)

# Check if either pystan or pystan3 is installed
pystan_installed = (importlib.util.find_spec("pystan") is not None) or (
    importlib.util.find_spec("stan") is not None
)


@pytest.mark.skipif(
    not (pystan_installed or "ARVIZ_REQUIRE_ALL_DEPS" in os.environ),
    reason="test requires pystan/pystan3 which is not installed",
)
class TestDataPyStan:
    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        class Data:
            model, obj = load_cached_models(eight_schools_params, draws, chains, "pystan")["pystan"]

        return Data

    def get_inference_data(self, data, eight_schools_params):
        """vars as str."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive="y_hat",
            predictions="y_hat",  # wrong, but fine for testing
            prior=data.obj,
            prior_predictive="y_hat",
            observed_data="y",
            constant_data="sigma",
            predictions_constant_data="sigma",  # wrong, but fine for testing
            log_likelihood={"y": "log_lik"},
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "sigma": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
            },
            posterior_model=data.model,
            prior_model=data.model,
        )

    def get_inference_data2(self, data, eight_schools_params):
        """vars as lists."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive=["y_hat"],
            predictions=["y_hat"],  # wrong, but fine for testing
            prior=data.obj,
            prior_predictive=["y_hat"],
            observed_data=["y"],
            log_likelihood="log_lik",
            coords={
                "school": np.arange(eight_schools_params["J"]),
                "log_likelihood_dim": np.arange(eight_schools_params["J"]),
            },
            dims={
                "theta": ["school"],
                "y": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
                "log_lik": ["log_likelihood_dim"],
            },
            posterior_model=data.model,
            prior_model=data.model,
        )

    def get_inference_data3(self, data, eight_schools_params):
        """multiple vars as lists."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive=["y_hat", "log_lik"],  # wrong, but fine for testing
            predictions=["y_hat", "log_lik"],  # wrong, but fine for testing
            prior=data.obj,
            prior_predictive=["y_hat", "log_lik"],  # wrong, but fine for testing
            constant_data=["sigma", "y"],  # wrong, but fine for testing
            predictions_constant_data=["sigma", "y"],  # wrong, but fine for testing
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "sigma": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
            },
            posterior_model=data.model,
            prior_model=data.model,
        )

    def get_inference_data4(self, data):
        """minimal input."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive=None,
            prior=data.obj,
            prior_predictive=None,
            coords=None,
            dims=None,
            posterior_model=data.model,
            log_likelihood=[],
            prior_model=data.model,
            save_warmup=True,
        )

    def get_inference_data5(self, data):
        """minimal input."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive=None,
            prior=data.obj,
            prior_predictive=None,
            coords=None,
            dims=None,
            posterior_model=data.model,
            log_likelihood=False,
            prior_model=data.model,
            save_warmup=True,
            dtypes={"eta": int},
        )

    def test_sampler_stats(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        test_dict = {"sample_stats": ["diverging"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_inference_data(self, data, eight_schools_params):
        inference_data1 = self.get_inference_data(data, eight_schools_params)
        inference_data2 = self.get_inference_data2(data, eight_schools_params)
        inference_data3 = self.get_inference_data3(data, eight_schools_params)
        inference_data4 = self.get_inference_data4(data)
        inference_data5 = self.get_inference_data5(data)
        # inference_data 1
        test_dict = {
            "posterior": ["theta", "~log_lik"],
            "posterior_predictive": ["y_hat"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "constant_data": ["sigma"],
            "predictions_constant_data": ["sigma"],
            "sample_stats": ["diverging", "lp"],
            "log_likelihood": ["y", "~log_lik"],
            "prior": ["theta"],
        }
        fails = check_multiple_attrs(test_dict, inference_data1)
        assert not fails
        # inference_data 2
        test_dict = {
            "posterior_predictive": ["y_hat"],
            "predictions": ["y_hat"],
            "observed_data": ["y"],
            "sample_stats_prior": ["diverging"],
            "sample_stats": ["diverging", "lp"],
            "log_likelihood": ["log_lik"],
            "prior_predictive": ["y_hat"],
        }
        fails = check_multiple_attrs(test_dict, inference_data2)
        assert not fails
        assert any(
            item in inference_data2.posterior.attrs for item in ["stan_code", "program_code"]
        )
        assert any(
            item in inference_data2.sample_stats.attrs for item in ["stan_code", "program_code"]
        )
        # inference_data 3
        test_dict = {
            "posterior_predictive": ["y_hat", "log_lik"],
            "predictions": ["y_hat", "log_lik"],
            "constant_data": ["sigma", "y"],
            "predictions_constant_data": ["sigma", "y"],
            "sample_stats_prior": ["diverging"],
            "sample_stats": ["diverging", "lp"],
            "log_likelihood": ["log_lik"],
            "prior_predictive": ["y_hat", "log_lik"],
        }
        fails = check_multiple_attrs(test_dict, inference_data3)
        assert not fails
        # inference_data 4
        test_dict = {
            "posterior": ["theta"],
            "prior": ["theta"],
            "sample_stats": ["diverging", "lp"],
            "~log_likelihood": [""],
            "warmup_posterior": ["theta"],
            "warmup_sample_stats": ["diverging", "lp"],
        }
        fails = check_multiple_attrs(test_dict, inference_data4)
        assert not fails
        # inference_data 5
        test_dict = {
            "posterior": ["theta"],
            "prior": ["theta"],
            "sample_stats": ["diverging", "lp"],
            "~log_likelihood": [""],
            "warmup_posterior": ["theta"],
            "warmup_sample_stats": ["diverging", "lp"],
        }
        fails = check_multiple_attrs(test_dict, inference_data5)
        assert not fails
        assert inference_data5.posterior.eta.dtype.kind == "i"

    def test_invalid_fit(self, data):
        if pystan_version() == 2:
            model = data.model
            model_data = {
                "J": 8,
                "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
                "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
            }
            fit_test_grad = model.sampling(
                data=model_data, test_grad=True, check_hmc_diagnostics=False
            )
            with pytest.raises(AttributeError):
                _ = from_pystan(posterior=fit_test_grad)
            fit = model.sampling(data=model_data, iter=100, chains=1, check_hmc_diagnostics=False)
            del fit.sim["samples"]
            with pytest.raises(AttributeError):
                _ = from_pystan(posterior=fit)

    def test_empty_parameter(self):
        model_code = """
            parameters {
                real y;
                vector[3] x;
                vector[0] a;
                vector[2] z;
            }
            model {
                y ~ normal(0,1);
            }
        """
        if pystan_version() == 2:
            from pystan import StanModel  # pylint: disable=import-error

            model = StanModel(model_code=model_code)
            fit = model.sampling(iter=500, chains=2, check_hmc_diagnostics=False)
        else:
            import stan  # pylint: disable=import-error

            model = stan.build(model_code)
            fit = model.sample(num_samples=500, num_chains=2)

        posterior = from_pystan(posterior=fit)
        test_dict = {"posterior": ["y", "x", "z", "~a"], "sample_stats": ["diverging"]}
        fails = check_multiple_attrs(test_dict, posterior)
        assert not fails

    def test_get_draws(self, data):
        fit = data.obj
        if pystan_version() == 2:
            draws, _ = get_draws(fit, variables=["theta", "theta"])
        else:
            draws, _ = get_draws_stan3(fit, variables=["theta", "theta"])
        assert draws.get("theta") is not None

    @pytest.mark.skipif(pystan_version() != 2, reason="PyStan 2.x required")
    def test_index_order(self, data, eight_schools_params):
        """Test 0-indexed data."""
        # Skip test if pystan not installed
        pystan = importorskip("pystan")  # pylint: disable=import-error

        fit = data.model.sampling(data=eight_schools_params)
        if pystan.__version__ >= "2.18":
            # make 1-indexed to 0-indexed
            for holder in fit.sim["samples"]:
                new_chains = OrderedDict()
                for i, (key, values) in enumerate(holder.chains.items()):
                    if "[" in key:
                        name, *shape = key.replace("]", "").split("[")
                        shape = [str(int(item) - 1) for items in shape for item in items.split(",")]
                        key = f"{name}[{','.join(shape)}]"
                    new_chains[key] = np.full_like(values, fill_value=float(i))
                setattr(holder, "chains", new_chains)
            fit.sim["fnames_oi"] = list(fit.sim["samples"][0].chains.keys())
        idata = from_pystan(posterior=fit)
        assert idata is not None
        for j, fpar in enumerate(fit.sim["fnames_oi"]):
            par, *shape = fpar.replace("]", "").split("[")
            if par in {"lp__", "log_lik"}:
                continue
            assert hasattr(idata.posterior, par), (par, list(idata.posterior.data_vars))
            if shape:
                shape = [slice(None), slice(None)] + list(map(int, shape))
                assert idata.posterior[par][tuple(shape)].values.mean() == float(j)
            else:
                assert idata.posterior[par].values.mean() == float(j)
