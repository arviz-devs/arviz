# pylint: disable=no-member, invalid-name, redefined-outer-name
from collections import OrderedDict
import numpy as np
import pytest

from arviz import from_pystan
from ..data.io_pystan import get_draws, get_draws_stan3  # pylint: disable=unused-import
from .helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    load_cached_models,
    pystan_version,
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
            prior=data.obj,
            prior_predictive="y_hat",
            observed_data="y",
            constant_data="sigma",
            log_likelihood="log_lik",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "sigma": ["school"],
                "log_lik": ["school"],
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
            posterior_predictive=["y_hat", "log_lik"],
            prior=data.obj,
            prior_predictive=["y_hat", "log_lik"],
            constant_data=["sigma"],
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
        """multiple vars as lists."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive=None,
            prior=data.obj,
            prior_predictive=None,
            coords=None,
            dims=None,
            posterior_model=data.model,
            prior_model=data.model,
        )

    def test_sampler_stats(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        test_dict = {"sample_stats": ["lp", "diverging"]}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_inference_data(self, data, eight_schools_params):
        inference_data1 = self.get_inference_data(data, eight_schools_params)
        inference_data2 = self.get_inference_data2(data, eight_schools_params)
        inference_data3 = self.get_inference_data3(data, eight_schools_params)
        inference_data4 = self.get_inference_data4(data)
        # inference_data 1
        test_dict = {
            "posterior": ["theta"],
            "observed_data": ["y"],
            "constant_data": ["sigma"],
            "sample_stats": ["log_likelihood"],
            "prior": ["theta"],
        }
        fails = check_multiple_attrs(test_dict, inference_data1)
        assert not fails
        # inference_data 2
        test_dict = {
            "posterior_predictive": ["y_hat"],
            "observed_data": ["y"],
            "sample_stats_prior": ["lp"],
            "sample_stats": ["lp"],
            "prior_predictive": ["y_hat"],
        }
        fails = check_multiple_attrs(test_dict, inference_data2)
        assert not fails
        # inference_data 3
        test_dict = {
            "posterior_predictive": ["y_hat"],
            "constant_data": ["sigma"],
            "sample_stats_prior": ["lp"],
            "sample_stats": ["lp"],
            "prior_predictive": ["y_hat"],
        }
        fails = check_multiple_attrs(test_dict, inference_data3)
        assert not fails
        # inference_data 4
        test_dict = {"posterior": ["theta"], "prior": ["theta"]}
        fails = check_multiple_attrs(test_dict, inference_data4)
        assert not fails

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
        if pystan_version() == 2:
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
            from pystan import StanModel

            model = StanModel(model_code=model_code)
            fit = model.sampling(iter=10, chains=2, check_hmc_diagnostics=False)
            posterior = from_pystan(posterior=fit)
            test_dict = {"posterior": ["y", "x", "z"], "sample_stats": ["lp"]}
            fails = check_multiple_attrs(test_dict, posterior)
            assert not fails

    def test_get_draws(self, data):
        fit = data.obj
        if pystan_version() == 2:
            draws = get_draws(fit, variables=["theta", "theta"])
            assert draws.get("theta") is not None
        else:
            draws = get_draws_stan3(fit, variables=["theta", "theta"])
            assert draws.get("theta") is not None

    @pytest.mark.skipif(pystan_version() != 2, reason="PyStan 2.x required")
    def test_index_order(self, data, eight_schools_params):
        """Test 0-indexed data."""
        import pystan

        fit = data.model.sampling(data=eight_schools_params)
        if pystan.__version__ >= "2.18":
            # make 1-indexed to 0-indexed
            for holder in fit.sim["samples"]:
                new_chains = OrderedDict()
                for i, (key, values) in enumerate(holder.chains.items()):
                    if "[" in key:
                        name, *shape = key.replace("]", "").split("[")
                        shape = [str(int(item) - 1) for items in shape for item in items.split(",")]
                        key = name + "[{}]".format(",".join(shape))
                    new_chains[key] = np.full_like(values, fill_value=float(i))
                setattr(holder, "chains", new_chains)
            fit.sim["fnames_oi"] = list(fit.sim["samples"][0].chains.keys())
        idata = from_pystan(posterior=fit)
        assert idata is not None
        for j, fpar in enumerate(fit.sim["fnames_oi"]):
            if fpar == "lp__":
                continue
            par, *shape = fpar.replace("]", "").split("[")
            assert hasattr(idata.posterior, par)
            if shape:
                shape = [slice(None), slice(None)] + list(map(int, shape))
                assert idata.posterior[par][tuple(shape)].values.mean() == float(j)
            else:
                assert idata.posterior[par].values.mean() == float(j)
