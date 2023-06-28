#  pylint: disable=too-many-instance-attributes,too-many-lines
"""PyStan-specific conversion code."""
import re
from collections import OrderedDict
from copy import deepcopy
from math import ceil

import numpy as np
import xarray as xr

from .. import _log
from ..rcparams import rcParams
from .base import dict_to_dataset, generate_dims_coords, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData

try:
    import ujson as json
except ImportError:
    # Can't find ujson using json
    # mypy struggles with conditional imports expressed as catching ImportError:
    # https://github.com/python/mypy/issues/1153
    import json  # type: ignore


class PyStanConverter:
    """Encapsulate PyStan specific logic."""

    def __init__(
        self,
        *,
        posterior=None,
        posterior_predictive=None,
        predictions=None,
        prior=None,
        prior_predictive=None,
        observed_data=None,
        constant_data=None,
        predictions_constant_data=None,
        log_likelihood=None,
        coords=None,
        dims=None,
        save_warmup=None,
        dtypes=None,
    ):
        self.posterior = posterior
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions
        self.prior = prior
        self.prior_predictive = prior_predictive
        self.observed_data = observed_data
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.log_likelihood = (
            rcParams["data.log_likelihood"] if log_likelihood is None else log_likelihood
        )
        self.coords = coords
        self.dims = dims
        self.save_warmup = rcParams["data.save_warmup"] if save_warmup is None else save_warmup
        self.dtypes = dtypes

        if (
            self.log_likelihood is True
            and self.posterior is not None
            and "log_lik" in self.posterior.sim["pars_oi"]
        ):
            self.log_likelihood = ["log_lik"]
        elif isinstance(self.log_likelihood, bool):
            self.log_likelihood = None

        import pystan  # pylint: disable=import-error

        self.pystan = pystan

    @requires("posterior")
    def posterior_to_xarray(self):
        """Extract posterior samples from fit."""
        posterior = self.posterior
        # filter posterior_predictive and log_likelihood
        posterior_predictive = self.posterior_predictive
        if posterior_predictive is None:
            posterior_predictive = []
        elif isinstance(posterior_predictive, str):
            posterior_predictive = [posterior_predictive]
        predictions = self.predictions
        if predictions is None:
            predictions = []
        elif isinstance(predictions, str):
            predictions = [predictions]
        log_likelihood = self.log_likelihood
        if log_likelihood is None:
            log_likelihood = []
        elif isinstance(log_likelihood, str):
            log_likelihood = [log_likelihood]
        elif isinstance(log_likelihood, dict):
            log_likelihood = list(log_likelihood.values())

        ignore = posterior_predictive + predictions + log_likelihood + ["lp__"]

        data, data_warmup = get_draws(
            posterior, ignore=ignore, warmup=self.save_warmup, dtypes=self.dtypes
        )
        attrs = get_attrs(posterior)
        return (
            dict_to_dataset(
                data, library=self.pystan, attrs=attrs, coords=self.coords, dims=self.dims
            ),
            dict_to_dataset(
                data_warmup, library=self.pystan, attrs=attrs, coords=self.coords, dims=self.dims
            ),
        )

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from posterior."""
        posterior = self.posterior

        data, data_warmup = get_sample_stats(posterior, warmup=self.save_warmup)

        # lp__
        stat_lp, stat_lp_warmup = get_draws(
            posterior, variables="lp__", warmup=self.save_warmup, dtypes=self.dtypes
        )
        data["lp"] = stat_lp["lp__"]
        if stat_lp_warmup:
            data_warmup["lp"] = stat_lp_warmup["lp__"]

        attrs = get_attrs(posterior)
        return (
            dict_to_dataset(
                data, library=self.pystan, attrs=attrs, coords=self.coords, dims=self.dims
            ),
            dict_to_dataset(
                data_warmup, library=self.pystan, attrs=attrs, coords=self.coords, dims=self.dims
            ),
        )

    @requires("posterior")
    @requires("log_likelihood")
    def log_likelihood_to_xarray(self):
        """Store log_likelihood data in log_likelihood group."""
        fit = self.posterior

        # log_likelihood values
        log_likelihood = self.log_likelihood
        if isinstance(log_likelihood, str):
            log_likelihood = [log_likelihood]
        if isinstance(log_likelihood, (list, tuple)):
            log_likelihood = {name: name for name in log_likelihood}
        log_likelihood_draws, log_likelihood_draws_warmup = get_draws(
            fit,
            variables=list(log_likelihood.values()),
            warmup=self.save_warmup,
            dtypes=self.dtypes,
        )
        data = {
            obs_var_name: log_likelihood_draws[log_like_name]
            for obs_var_name, log_like_name in log_likelihood.items()
            if log_like_name in log_likelihood_draws
        }

        data_warmup = {
            obs_var_name: log_likelihood_draws_warmup[log_like_name]
            for obs_var_name, log_like_name in log_likelihood.items()
            if log_like_name in log_likelihood_draws_warmup
        }

        return (
            dict_to_dataset(
                data, library=self.pystan, coords=self.coords, dims=self.dims, skip_event_dims=True
            ),
            dict_to_dataset(
                data_warmup,
                library=self.pystan,
                coords=self.coords,
                dims=self.dims,
                skip_event_dims=True,
            ),
        )

    @requires("posterior")
    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        posterior = self.posterior
        posterior_predictive = self.posterior_predictive
        data, data_warmup = get_draws(
            posterior, variables=posterior_predictive, warmup=self.save_warmup, dtypes=self.dtypes
        )
        return (
            dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims),
            dict_to_dataset(data_warmup, library=self.pystan, coords=self.coords, dims=self.dims),
        )

    @requires("posterior")
    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert predictions samples to xarray."""
        posterior = self.posterior
        predictions = self.predictions
        data, data_warmup = get_draws(
            posterior, variables=predictions, warmup=self.save_warmup, dtypes=self.dtypes
        )
        return (
            dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims),
            dict_to_dataset(data_warmup, library=self.pystan, coords=self.coords, dims=self.dims),
        )

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        prior = self.prior
        # filter posterior_predictive and log_likelihood
        prior_predictive = self.prior_predictive
        if prior_predictive is None:
            prior_predictive = []
        elif isinstance(prior_predictive, str):
            prior_predictive = [prior_predictive]

        ignore = prior_predictive + ["lp__"]

        data, _ = get_draws(prior, ignore=ignore, warmup=False, dtypes=self.dtypes)
        attrs = get_attrs(prior)
        return dict_to_dataset(
            data, library=self.pystan, attrs=attrs, coords=self.coords, dims=self.dims
        )

    @requires("prior")
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats_prior from prior."""
        prior = self.prior
        data, _ = get_sample_stats(prior, warmup=False)

        # lp__
        stat_lp, _ = get_draws(prior, variables="lp__", warmup=False, dtypes=self.dtypes)
        data["lp"] = stat_lp["lp__"]

        attrs = get_attrs(prior)
        return dict_to_dataset(
            data, library=self.pystan, attrs=attrs, coords=self.coords, dims=self.dims
        )

    @requires("prior")
    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        prior = self.prior
        prior_predictive = self.prior_predictive
        data, _ = get_draws(prior, variables=prior_predictive, warmup=False, dtypes=self.dtypes)
        return dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims)

    @requires("posterior")
    @requires(["observed_data", "constant_data", "predictions_constant_data"])
    def data_to_xarray(self):
        """Convert observed, constant data and predictions constant data to xarray."""
        posterior = self.posterior
        dims = {} if self.dims is None else self.dims
        obs_const_dict = {}
        for group_name in ("observed_data", "constant_data", "predictions_constant_data"):
            names = getattr(self, group_name)
            if names is None:
                continue
            names = [names] if isinstance(names, str) else names
            data = OrderedDict()
            for key in names:
                vals = np.atleast_1d(posterior.data[key])
                val_dims = dims.get(key)
                val_dims, coords = generate_dims_coords(
                    vals.shape, key, dims=val_dims, coords=self.coords
                )
                data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
            obs_const_dict[group_name] = xr.Dataset(
                data_vars=data, attrs=make_attrs(library=self.pystan)
            )
        return obs_const_dict

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `fit`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        data_dict = self.data_to_xarray()
        return InferenceData(
            save_warmup=self.save_warmup,
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                "log_likelihood": self.log_likelihood_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "predictions": self.predictions_to_xarray(),
                "prior": self.prior_to_xarray(),
                "sample_stats_prior": self.sample_stats_prior_to_xarray(),
                "prior_predictive": self.prior_predictive_to_xarray(),
                **({} if data_dict is None else data_dict),
            },
        )


class PyStan3Converter:
    """Encapsulate PyStan3 specific logic."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        *,
        posterior=None,
        posterior_model=None,
        posterior_predictive=None,
        predictions=None,
        prior=None,
        prior_model=None,
        prior_predictive=None,
        observed_data=None,
        constant_data=None,
        predictions_constant_data=None,
        log_likelihood=None,
        coords=None,
        dims=None,
        save_warmup=None,
        dtypes=None,
    ):
        self.posterior = posterior
        self.posterior_model = posterior_model
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions
        self.prior = prior
        self.prior_model = prior_model
        self.prior_predictive = prior_predictive
        self.observed_data = observed_data
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.log_likelihood = (
            rcParams["data.log_likelihood"] if log_likelihood is None else log_likelihood
        )
        self.coords = coords
        self.dims = dims
        self.save_warmup = rcParams["data.save_warmup"] if save_warmup is None else save_warmup
        self.dtypes = dtypes

        if (
            self.log_likelihood is True
            and self.posterior is not None
            and "log_lik" in self.posterior.param_names
        ):
            self.log_likelihood = ["log_lik"]
        elif isinstance(self.log_likelihood, bool):
            self.log_likelihood = None

        import stan  # pylint: disable=import-error

        self.stan = stan

    @requires("posterior")
    def posterior_to_xarray(self):
        """Extract posterior samples from fit."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        # filter posterior_predictive and log_likelihood
        posterior_predictive = self.posterior_predictive
        if posterior_predictive is None:
            posterior_predictive = []
        elif isinstance(posterior_predictive, str):
            posterior_predictive = [posterior_predictive]
        predictions = self.predictions
        if predictions is None:
            predictions = []
        elif isinstance(predictions, str):
            predictions = [predictions]
        log_likelihood = self.log_likelihood
        if log_likelihood is None:
            log_likelihood = []
        elif isinstance(log_likelihood, str):
            log_likelihood = [log_likelihood]
        elif isinstance(log_likelihood, dict):
            log_likelihood = list(log_likelihood.values())

        ignore = posterior_predictive + predictions + log_likelihood

        data, data_warmup = get_draws_stan3(
            posterior,
            model=posterior_model,
            ignore=ignore,
            warmup=self.save_warmup,
            dtypes=self.dtypes,
        )
        attrs = get_attrs_stan3(posterior, model=posterior_model)
        return (
            dict_to_dataset(
                data, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims
            ),
            dict_to_dataset(
                data_warmup, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims
            ),
        )

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from posterior."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        data, data_warmup = get_sample_stats_stan3(
            posterior, ignore="lp__", warmup=self.save_warmup, dtypes=self.dtypes
        )
        data_lp, data_warmup_lp = get_sample_stats_stan3(
            posterior, variables="lp__", warmup=self.save_warmup
        )
        data["lp"] = data_lp["lp"]
        if data_warmup_lp:
            data_warmup["lp"] = data_warmup_lp["lp"]

        attrs = get_attrs_stan3(posterior, model=posterior_model)
        return (
            dict_to_dataset(
                data, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims
            ),
            dict_to_dataset(
                data_warmup, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims
            ),
        )

    @requires("posterior")
    @requires("log_likelihood")
    def log_likelihood_to_xarray(self):
        """Store log_likelihood data in log_likelihood group."""
        fit = self.posterior

        log_likelihood = self.log_likelihood
        model = self.posterior_model
        if isinstance(log_likelihood, str):
            log_likelihood = [log_likelihood]
        if isinstance(log_likelihood, (list, tuple)):
            log_likelihood = {name: name for name in log_likelihood}
        log_likelihood_draws, log_likelihood_draws_warmup = get_draws_stan3(
            fit,
            model=model,
            variables=list(log_likelihood.values()),
            warmup=self.save_warmup,
            dtypes=self.dtypes,
        )
        data = {
            obs_var_name: log_likelihood_draws[log_like_name]
            for obs_var_name, log_like_name in log_likelihood.items()
            if log_like_name in log_likelihood_draws
        }
        data_warmup = {
            obs_var_name: log_likelihood_draws_warmup[log_like_name]
            for obs_var_name, log_like_name in log_likelihood.items()
            if log_like_name in log_likelihood_draws_warmup
        }

        return (
            dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims),
            dict_to_dataset(data_warmup, library=self.stan, coords=self.coords, dims=self.dims),
        )

    @requires("posterior")
    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        posterior_predictive = self.posterior_predictive
        data, data_warmup = get_draws_stan3(
            posterior,
            model=posterior_model,
            variables=posterior_predictive,
            warmup=self.save_warmup,
            dtypes=self.dtypes,
        )
        return (
            dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims),
            dict_to_dataset(data_warmup, library=self.stan, coords=self.coords, dims=self.dims),
        )

    @requires("posterior")
    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert predictions samples to xarray."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        predictions = self.predictions
        data, data_warmup = get_draws_stan3(
            posterior,
            model=posterior_model,
            variables=predictions,
            warmup=self.save_warmup,
            dtypes=self.dtypes,
        )
        return (
            dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims),
            dict_to_dataset(data_warmup, library=self.stan, coords=self.coords, dims=self.dims),
        )

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        prior = self.prior
        prior_model = self.prior_model
        # filter posterior_predictive and log_likelihood
        prior_predictive = self.prior_predictive
        if prior_predictive is None:
            prior_predictive = []
        elif isinstance(prior_predictive, str):
            prior_predictive = [prior_predictive]

        ignore = prior_predictive

        data, data_warmup = get_draws_stan3(
            prior, model=prior_model, ignore=ignore, warmup=self.save_warmup, dtypes=self.dtypes
        )
        attrs = get_attrs_stan3(prior, model=prior_model)
        return (
            dict_to_dataset(
                data, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims
            ),
            dict_to_dataset(
                data_warmup, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims
            ),
        )

    @requires("prior")
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats_prior from prior."""
        prior = self.prior
        prior_model = self.prior_model
        data, data_warmup = get_sample_stats_stan3(
            prior, warmup=self.save_warmup, dtypes=self.dtypes
        )
        attrs = get_attrs_stan3(prior, model=prior_model)
        return (
            dict_to_dataset(
                data, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims
            ),
            dict_to_dataset(
                data_warmup, library=self.stan, attrs=attrs, coords=self.coords, dims=self.dims
            ),
        )

    @requires("prior")
    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        prior = self.prior
        prior_model = self.prior_model
        prior_predictive = self.prior_predictive
        data, data_warmup = get_draws_stan3(
            prior,
            model=prior_model,
            variables=prior_predictive,
            warmup=self.save_warmup,
            dtypes=self.dtypes,
        )
        return (
            dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims),
            dict_to_dataset(data_warmup, library=self.stan, coords=self.coords, dims=self.dims),
        )

    @requires("posterior_model")
    @requires(["observed_data", "constant_data"])
    def observed_and_constant_data_to_xarray(self):
        """Convert observed data to xarray."""
        posterior_model = self.posterior_model
        dims = {} if self.dims is None else self.dims
        obs_const_dict = {}
        for group_name in ("observed_data", "constant_data"):
            names = getattr(self, group_name)
            if names is None:
                continue
            names = [names] if isinstance(names, str) else names
            data = OrderedDict()
            for key in names:
                vals = np.atleast_1d(posterior_model.data[key])
                val_dims = dims.get(key)
                val_dims, coords = generate_dims_coords(
                    vals.shape, key, dims=val_dims, coords=self.coords
                )
                data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
            obs_const_dict[group_name] = xr.Dataset(
                data_vars=data, attrs=make_attrs(library=self.stan)
            )
        return obs_const_dict

    @requires("posterior_model")
    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert observed data to xarray."""
        posterior_model = self.posterior_model
        dims = {} if self.dims is None else self.dims
        names = self.predictions_constant_data
        names = [names] if isinstance(names, str) else names
        data = OrderedDict()
        for key in names:
            vals = np.atleast_1d(posterior_model.data[key])
            val_dims = dims.get(key)
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=data, attrs=make_attrs(library=self.stan))

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `fit`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        obs_const_dict = self.observed_and_constant_data_to_xarray()
        predictions_const_data = self.predictions_constant_data_to_xarray()
        return InferenceData(
            save_warmup=self.save_warmup,
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                "log_likelihood": self.log_likelihood_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "predictions": self.predictions_to_xarray(),
                "prior": self.prior_to_xarray(),
                "sample_stats_prior": self.sample_stats_prior_to_xarray(),
                "prior_predictive": self.prior_predictive_to_xarray(),
                **({} if obs_const_dict is None else obs_const_dict),
                **(
                    {}
                    if predictions_const_data is None
                    else {"predictions_constant_data": predictions_const_data}
                ),
            },
        )


def get_draws(fit, variables=None, ignore=None, warmup=False, dtypes=None):
    """Extract draws from PyStan fit."""
    if ignore is None:
        ignore = []
    if fit.mode == 1:
        msg = "Model in mode 'test_grad'. Sampling is not conducted."
        raise AttributeError(msg)

    if fit.mode == 2 or fit.sim.get("samples") is None:
        msg = "Fit doesn't contain samples."
        raise AttributeError(msg)

    if dtypes is None:
        dtypes = {}

    dtypes = {**infer_dtypes(fit), **dtypes}

    if variables is None:
        variables = fit.sim["pars_oi"]
    elif isinstance(variables, str):
        variables = [variables]
    variables = list(variables)

    for var, dim in zip(fit.sim["pars_oi"], fit.sim["dims_oi"]):
        if var in variables and np.prod(dim) == 0:
            del variables[variables.index(var)]

    ndraws_warmup = fit.sim["warmup2"]
    if max(ndraws_warmup) == 0:
        warmup = False
    ndraws = [s - w for s, w in zip(fit.sim["n_save"], ndraws_warmup)]
    nchain = len(fit.sim["samples"])

    # check if the values are in 0-based (<=2.17) or 1-based indexing (>=2.18)
    shift = 1
    if any(dim and np.prod(dim) != 0 for dim in fit.sim["dims_oi"]):
        # choose variable with lowest number of dims > 1
        par_idx = min(
            (dim, i) for i, dim in enumerate(fit.sim["dims_oi"]) if (dim and np.prod(dim) != 0)
        )[1]
        offset = int(sum(map(np.prod, fit.sim["dims_oi"][:par_idx])))
        par_offset = int(np.prod(fit.sim["dims_oi"][par_idx]))
        par_keys = fit.sim["fnames_oi"][offset : offset + par_offset]
        shift = len(par_keys)
        for item in par_keys:
            _, shape = item.replace("]", "").split("[")
            shape_idx_min = min(int(shape_value) for shape_value in shape.split(","))
            if shape_idx_min < shift:
                shift = shape_idx_min
        # If shift is higher than 1, this will probably mean that Stan
        # has implemented sparse structure (saves only non-zero parts),
        # but let's hope that dims are still corresponding to the full shape
        shift = int(min(shift, 1))

    var_keys = OrderedDict((var, []) for var in fit.sim["pars_oi"])
    for key in fit.sim["fnames_oi"]:
        var, *tails = key.split("[")
        loc = [Ellipsis]
        for tail in tails:
            loc = []
            for i in tail[:-1].split(","):
                loc.append(int(i) - shift)
        var_keys[var].append((key, loc))

    shapes = dict(zip(fit.sim["pars_oi"], fit.sim["dims_oi"]))

    variables = [var for var in variables if var not in ignore]

    data = OrderedDict()
    data_warmup = OrderedDict()

    for var in variables:
        if var in data:
            continue
        keys_locs = var_keys.get(var, [(var, [Ellipsis])])
        shape = shapes.get(var, [])
        dtype = dtypes.get(var)

        ndraw = max(ndraws)
        ary_shape = [nchain, ndraw] + shape
        ary = np.empty(ary_shape, dtype=dtype, order="F")

        if warmup:
            nwarmup = max(ndraws_warmup)
            ary_warmup_shape = [nchain, nwarmup] + shape
            ary_warmup = np.empty(ary_warmup_shape, dtype=dtype, order="F")

        for chain, (pyholder, ndraw, ndraw_warmup) in enumerate(
            zip(fit.sim["samples"], ndraws, ndraws_warmup)
        ):
            axes = [chain, slice(None)]
            for key, loc in keys_locs:
                ary_slice = tuple(axes + loc)
                ary[ary_slice] = pyholder.chains[key][-ndraw:]
                if warmup:
                    ary_warmup[ary_slice] = pyholder.chains[key][:ndraw_warmup]
        data[var] = ary
        if warmup:
            data_warmup[var] = ary_warmup
    return data, data_warmup


def get_sample_stats(fit, warmup=False, dtypes=None):
    """Extract sample stats from PyStan fit."""
    if dtypes is None:
        dtypes = {}
    dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64, **dtypes}

    rename_dict = {
        "divergent": "diverging",
        "n_leapfrog": "n_steps",
        "treedepth": "tree_depth",
        "stepsize": "step_size",
        "accept_stat": "acceptance_rate",
    }

    ndraws_warmup = fit.sim["warmup2"]
    if max(ndraws_warmup) == 0:
        warmup = False
    ndraws = [s - w for s, w in zip(fit.sim["n_save"], ndraws_warmup)]

    extraction = OrderedDict()
    extraction_warmup = OrderedDict()
    for chain, (pyholder, ndraw, ndraw_warmup) in enumerate(
        zip(fit.sim["samples"], ndraws, ndraws_warmup)
    ):
        if chain == 0:
            for key in pyholder["sampler_param_names"]:
                extraction[key] = []
                if warmup:
                    extraction_warmup[key] = []
        for key, values in zip(pyholder["sampler_param_names"], pyholder["sampler_params"]):
            extraction[key].append(values[-ndraw:])
            if warmup:
                extraction_warmup[key].append(values[:ndraw_warmup])

    data = OrderedDict()
    for key, values in extraction.items():
        values = np.stack(values, axis=0)
        dtype = dtypes.get(key)
        values = values.astype(dtype)
        name = re.sub("__$", "", key)
        name = rename_dict.get(name, name)
        data[name] = values

    data_warmup = OrderedDict()
    if warmup:
        for key, values in extraction_warmup.items():
            values = np.stack(values, axis=0)
            values = values.astype(dtypes.get(key))
            name = re.sub("__$", "", key)
            name = rename_dict.get(name, name)
            data_warmup[name] = values

    return data, data_warmup


def get_attrs(fit):
    """Get attributes from PyStan fit object."""
    attrs = {}

    try:
        attrs["args"] = [deepcopy(holder.args) for holder in fit.sim["samples"]]
    except Exception as exp:  # pylint: disable=broad-except
        _log.warning("Failed to fetch args from fit: %s", exp)
    if "args" in attrs:
        for arg in attrs["args"]:
            if isinstance(arg["init"], bytes):
                arg["init"] = arg["init"].decode("utf-8")
        attrs["args"] = json.dumps(attrs["args"])
    try:
        attrs["inits"] = [holder.inits for holder in fit.sim["samples"]]
    except Exception as exp:  # pylint: disable=broad-except
        _log.warning("Failed to fetch `args` from fit: %s", exp)
    else:
        attrs["inits"] = json.dumps(attrs["inits"])

    attrs["step_size"] = []
    attrs["metric"] = []
    attrs["inv_metric"] = []
    for holder in fit.sim["samples"]:
        try:
            step_size = float(
                re.search(
                    r"step\s*size\s*=\s*([0-9]+.?[0-9]+)\s*",
                    holder.adaptation_info,
                    flags=re.IGNORECASE,
                ).group(1)
            )
        except AttributeError:
            step_size = np.nan
        attrs["step_size"].append(step_size)

        inv_metric_match = re.search(
            r"mass matrix:\s*(.*)\s*$", holder.adaptation_info, flags=re.DOTALL
        )
        if inv_metric_match:
            inv_metric_str = inv_metric_match.group(1)
            if "Diagonal elements of inverse mass matrix" in holder.adaptation_info:
                metric = "diag_e"
                inv_metric = [float(item) for item in inv_metric_str.strip(" #\n").split(",")]
            else:
                metric = "dense_e"
                inv_metric = [
                    list(map(float, item.split(",")))
                    for item in re.sub(r"#\s", "", inv_metric_str).splitlines()
                ]
        else:
            metric = "unit_e"
            inv_metric = None

        attrs["metric"].append(metric)
        attrs["inv_metric"].append(inv_metric)
    attrs["inv_metric"] = json.dumps(attrs["inv_metric"])

    if not attrs["step_size"]:
        del attrs["step_size"]

    attrs["adaptation_info"] = fit.get_adaptation_info()
    attrs["stan_code"] = fit.get_stancode()

    return attrs


def get_draws_stan3(fit, model=None, variables=None, ignore=None, warmup=False, dtypes=None):
    """Extract draws from PyStan3 fit."""
    if ignore is None:
        ignore = []

    if dtypes is None:
        dtypes = {}

    if model is not None:
        dtypes = {**infer_dtypes(fit, model), **dtypes}

    if not fit.save_warmup:
        warmup = False

    num_warmup = ceil((fit.num_warmup * fit.save_warmup) / fit.num_thin)

    if variables is None:
        variables = fit.param_names
    elif isinstance(variables, str):
        variables = [variables]
    variables = list(variables)

    data = OrderedDict()
    data_warmup = OrderedDict()

    for var in variables:
        if var in ignore:
            continue
        if var in data:
            continue
        dtype = dtypes.get(var)

        new_shape = (*fit.dims[fit.param_names.index(var)], -1, fit.num_chains)
        if 0 in new_shape:
            continue
        values = fit._draws[fit._parameter_indexes(var), :]  # pylint: disable=protected-access
        values = values.reshape(new_shape, order="F")
        values = np.moveaxis(values, [-2, -1], [1, 0])
        values = values.astype(dtype)
        if warmup:
            data_warmup[var] = values[:, num_warmup:]
        data[var] = values[:, num_warmup:]

    return data, data_warmup


def get_sample_stats_stan3(fit, variables=None, ignore=None, warmup=False, dtypes=None):
    """Extract sample stats from PyStan3 fit."""
    if dtypes is None:
        dtypes = {}
    dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64, **dtypes}

    rename_dict = {
        "divergent": "diverging",
        "n_leapfrog": "n_steps",
        "treedepth": "tree_depth",
        "stepsize": "step_size",
        "accept_stat": "acceptance_rate",
    }

    if isinstance(variables, str):
        variables = [variables]
    if isinstance(ignore, str):
        ignore = [ignore]

    if not fit.save_warmup:
        warmup = False

    num_warmup = ceil((fit.num_warmup * fit.save_warmup) / fit.num_thin)

    data = OrderedDict()
    data_warmup = OrderedDict()
    for key in fit.sample_and_sampler_param_names:
        if (variables and key not in variables) or (ignore and key in ignore):
            continue
        new_shape = -1, fit.num_chains
        values = fit._draws[fit._parameter_indexes(key)]  # pylint: disable=protected-access
        values = values.reshape(new_shape, order="F")
        values = np.moveaxis(values, [-2, -1], [1, 0])
        dtype = dtypes.get(key)
        values = values.astype(dtype)
        name = re.sub("__$", "", key)
        name = rename_dict.get(name, name)
        if warmup:
            data_warmup[name] = values[:, :num_warmup]
        data[name] = values[:, num_warmup:]

    return data, data_warmup


def get_attrs_stan3(fit, model=None):
    """Get attributes from PyStan3 fit and model object."""
    attrs = {}
    for key in ["num_chains", "num_samples", "num_thin", "num_warmup", "save_warmup"]:
        try:
            attrs[key] = getattr(fit, key)
        except AttributeError as exp:
            _log.warning("Failed to access attribute %s in fit object %s", key, exp)

    if model is not None:
        for key in ["model_name", "program_code", "random_seed"]:
            try:
                attrs[key] = getattr(model, key)
            except AttributeError as exp:
                _log.warning("Failed to access attribute %s in model object %s", key, exp)

    return attrs


def infer_dtypes(fit, model=None):
    """Infer dtypes from Stan model code.

    Function strips out generated quantities block and searches for `int`
    dtypes after stripping out comments inside the block.
    """
    if model is None:
        stan_code = fit.get_stancode()
        model_pars = fit.model_pars
    else:
        stan_code = model.program_code
        model_pars = fit.param_names

    dtypes = {key: item for key, item in infer_stan_dtypes(stan_code).items() if key in model_pars}
    return dtypes


# pylint disable=too-many-instance-attributes
def from_pystan(
    posterior=None,
    *,
    posterior_predictive=None,
    predictions=None,
    prior=None,
    prior_predictive=None,
    observed_data=None,
    constant_data=None,
    predictions_constant_data=None,
    log_likelihood=None,
    coords=None,
    dims=None,
    posterior_model=None,
    prior_model=None,
    save_warmup=None,
    dtypes=None,
):
    """Convert PyStan data into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_pystan <creating_InferenceData>`

    Parameters
    ----------
    posterior : StanFit4Model or stan.fit.Fit
        PyStan fit object for posterior.
    posterior_predictive : str, a list of str
        Posterior predictive samples for the posterior.
    predictions : str, a list of str
        Out-of-sample predictions for the posterior.
    prior : StanFit4Model or stan.fit.Fit
        PyStan fit object for prior.
    prior_predictive : str, a list of str
        Posterior predictive samples for the prior.
    observed_data : str or a list of str
        observed data used in the sampling.
        Observed data is extracted from the `posterior.data`.
        PyStan3 needs model object for the extraction.
        See `posterior_model`.
    constant_data : str or list of str
        Constants relevant to the model (i.e. x values in a linear
        regression).
    predictions_constant_data : str or list of str
        Constants relevant to the model predictions (i.e. new x values in a linear
        regression).
    log_likelihood : dict of {str: str}, list of str or str, optional
        Pointwise log_likelihood for the data. log_likelihood is extracted from the
        posterior. It is recommended to use this argument as a dictionary whose keys
        are observed variable names and its values are the variables storing log
        likelihood arrays in the Stan code. In other cases, a dictionary with keys
        equal to its values is used. By default, if a variable ``log_lik`` is
        present in the Stan model, it will be retrieved as pointwise log
        likelihood values. Use ``False`` or set ``data.log_likelihood`` to
        false to avoid this behaviour.
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable.
    posterior_model : stan.model.Model
        PyStan3 specific model object. Needed for automatic dtype parsing
        and for the extraction of observed data.
    prior_model : stan.model.Model
        PyStan3 specific model object. Needed for automatic dtype parsing.
    save_warmup : bool
        Save warmup iterations into InferenceData object. If not defined, use default
        defined by the rcParams.
    dtypes: dict
        A dictionary containing dtype information (int, float) for parameters.
        By default dtype information is extracted from the model code.
        Model code is extracted from fit object in PyStan 2 and from model object
        in PyStan 3.

    Returns
    -------
    InferenceData object
    """
    check_posterior = (posterior is not None) and (type(posterior).__module__ == "stan.fit")
    check_prior = (prior is not None) and (type(prior).__module__ == "stan.fit")
    if check_posterior or check_prior:
        return PyStan3Converter(
            posterior=posterior,
            posterior_model=posterior_model,
            posterior_predictive=posterior_predictive,
            predictions=predictions,
            prior=prior,
            prior_model=prior_model,
            prior_predictive=prior_predictive,
            observed_data=observed_data,
            constant_data=constant_data,
            predictions_constant_data=predictions_constant_data,
            log_likelihood=log_likelihood,
            coords=coords,
            dims=dims,
            save_warmup=save_warmup,
            dtypes=dtypes,
        ).to_inference_data()
    else:
        return PyStanConverter(
            posterior=posterior,
            posterior_predictive=posterior_predictive,
            predictions=predictions,
            prior=prior,
            prior_predictive=prior_predictive,
            observed_data=observed_data,
            constant_data=constant_data,
            predictions_constant_data=predictions_constant_data,
            log_likelihood=log_likelihood,
            coords=coords,
            dims=dims,
            save_warmup=save_warmup,
            dtypes=dtypes,
        ).to_inference_data()
