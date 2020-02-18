#  pylint: disable=too-many-instance-attributes
"""PyStan-specific conversion code."""
from collections import OrderedDict
import re

import numpy as np
import xarray as xr

from .inference_data import InferenceData
from .base import requires, dict_to_dataset, generate_dims_coords, make_attrs


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
        dims=None
    ):
        self.posterior = posterior
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions
        self.prior = prior
        self.prior_predictive = prior_predictive
        self.observed_data = observed_data
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.log_likelihood = log_likelihood
        self.coords = coords
        self.dims = dims

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

        data = get_draws(posterior, ignore=ignore)

        return dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims)

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from posterior."""
        posterior = self.posterior

        data = get_sample_stats(posterior)

        # lp__
        stat_lp = get_draws(posterior, variables="lp__")
        data["lp"] = stat_lp["lp__"]

        return dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims)

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
        log_likelihood_draws = get_draws(fit, variables=list(log_likelihood.values()))
        data = {
            obs_var_name: log_likelihood_draws[log_like_name]
            for obs_var_name, log_like_name in log_likelihood.items()
            if log_like_name in log_likelihood_draws
        }

        return dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims)

    @requires("posterior")
    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        posterior = self.posterior
        posterior_predictive = self.posterior_predictive
        data = get_draws(posterior, variables=posterior_predictive)
        return dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims)

    @requires("posterior")
    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert predictions samples to xarray."""
        posterior = self.posterior
        predictions = self.predictions
        data = get_draws(posterior, variables=predictions)
        return dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims)

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

        data = get_draws(prior, ignore=ignore)
        return dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims)

    @requires("prior")
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats_prior from prior."""
        prior = self.prior
        data = get_sample_stats(prior)
        return dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims)

    @requires("prior")
    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        prior = self.prior
        prior_predictive = self.prior_predictive
        data = get_draws(prior, variables=prior_predictive)
        return dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims)

    @requires("posterior")
    @requires(["observed_data", "constant_data"])
    def observed_and_constant_data_to_xarray(self):
        """Convert observed and constant data to xarray."""
        posterior = self.posterior
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
        obs_const_dict = {}
        for group_name in ("observed_data", "constant_data"):
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

    @requires("posterior")
    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert predictions constant data to xarray."""
        posterior = self.posterior
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
        names = self.predictions_constant_data
        names = [names] if isinstance(names, str) else names
        data = OrderedDict()
        for key in names:
            vals = np.atleast_1d(posterior.data[key])
            val_dims = dims.get(key)
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=data, attrs=make_attrs(library=self.pystan))

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `fit`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        obs_const_dict = self.observed_and_constant_data_to_xarray()
        predictions_const_data = self.predictions_constant_data_to_xarray()
        return InferenceData(
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
            }
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
        dims=None
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
        self.log_likelihood = log_likelihood
        self.coords = coords
        self.dims = dims

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

        data = get_draws_stan3(posterior, model=posterior_model, ignore=ignore)

        return dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims)

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from posterior."""
        posterior = self.posterior
        data = get_sample_stats_stan3(posterior, ignore="lp__")
        data["lp"] = get_sample_stats_stan3(posterior, variables="lp__")["lp"]
        return dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims)

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
        log_likelihood_draws = get_draws_stan3(
            fit, model=model, variables=list(log_likelihood.values())
        )
        data = {
            obs_var_name: log_likelihood_draws[log_like_name]
            for obs_var_name, log_like_name in log_likelihood.items()
            if log_like_name in log_likelihood_draws
        }

        return dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims)

    @requires("posterior")
    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        posterior_predictive = self.posterior_predictive
        data = get_draws_stan3(posterior, model=posterior_model, variables=posterior_predictive)
        return dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims)

    @requires("posterior")
    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert predictions samples to xarray."""
        posterior = self.posterior
        posterior_model = self.posterior_model
        predictions = self.predictions
        data = get_draws_stan3(posterior, model=posterior_model, variables=predictions)
        return dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims)

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

        data = get_draws_stan3(prior, model=prior_model, ignore=ignore)
        return dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims)

    @requires("prior")
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats_prior from prior."""
        prior = self.prior
        data = get_sample_stats_stan3(prior)
        return dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims)

    @requires("prior")
    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        prior = self.prior
        prior_model = self.prior_model
        prior_predictive = self.prior_predictive
        data = get_draws_stan3(prior, model=prior_model, variables=prior_predictive)
        return dict_to_dataset(data, library=self.stan, coords=self.coords, dims=self.dims)

    @requires("posterior_model")
    @requires(["observed_data", "constant_data"])
    def observed_and_constant_data_to_xarray(self):
        """Convert observed data to xarray."""
        posterior_model = self.posterior_model
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
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
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
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
            }
        )


def get_draws(fit, variables=None, ignore=None):
    """Extract draws from PyStan fit."""
    if ignore is None:
        ignore = []
    if fit.mode == 1:
        msg = "Model in mode 'test_grad'. Sampling is not conducted."
        raise AttributeError(msg)

    if fit.mode == 2 or fit.sim.get("samples") is None:
        msg = "Fit doesn't contain samples."
        raise AttributeError(msg)

    dtypes = infer_dtypes(fit)

    if variables is None:
        variables = fit.sim["pars_oi"]
    elif isinstance(variables, str):
        variables = [variables]
    variables = list(variables)

    for var, dim in zip(fit.sim["pars_oi"], fit.sim["dims_oi"]):
        if var in variables and np.prod(dim) == 0:
            del variables[variables.index(var)]

    ndraws = [s - w for s, w in zip(fit.sim["n_save"], fit.sim["warmup2"])]
    nchain = len(fit.sim["samples"])

    # check if the values are in 0-based (<=2.17) or 1-based indexing (>=2.18)
    shift = 1
    if any(dim and np.prod(dim) != 0 for dim in fit.sim["dims_oi"]):
        # choose variable with lowest number of dims > 1
        par_idx = min(
            (dim, i) for i, dim in enumerate(fit.sim["dims_oi"]) if (dim and np.prod(dim) != 0)
        )[1]
        offset = int(sum(map(np.product, fit.sim["dims_oi"][:par_idx])))
        par_offset = int(np.product(fit.sim["dims_oi"][par_idx]))
        par_keys = fit.sim["fnames_oi"][offset : offset + par_offset]
        shift = len(par_keys)
        for item in par_keys:
            _, shape = item.replace("]", "").split("[")
            shape_idx_min = min(int(shape_value) for shape_value in shape.split(","))
            if shape_idx_min < shift:
                shift = shape_idx_min
        # If shift is higher than 1, this will probably mean that Stan
        # has implemented sparse structure (saves only non-zero parts),
        # but let's hope that dims are still corresponding the full shape
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

    for var in variables:
        if var in data:
            continue
        keys_locs = var_keys.get(var, [(var, [Ellipsis])])
        shape = shapes.get(var, [])
        dtype = dtypes.get(var)

        ndraw = max(ndraws)
        ary_shape = [nchain, ndraw] + shape
        ary = np.empty(ary_shape, dtype=dtype, order="F")
        for chain, (pyholder, ndraw) in enumerate(zip(fit.sim["samples"], ndraws)):
            axes = [chain, slice(None)]
            for key, loc in keys_locs:
                ary_slice = tuple(axes + loc)
                ary[ary_slice] = pyholder.chains[key][-ndraw:]
        data[var] = ary

    return data


def get_sample_stats(fit):
    """Extract sample stats from PyStan fit."""
    dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64}

    ndraws = [s - w for s, w in zip(fit.sim["n_save"], fit.sim["warmup2"])]

    extraction = OrderedDict()
    for chain, (pyholder, ndraws) in enumerate(zip(fit.sim["samples"], ndraws)):
        if chain == 0:
            for key in pyholder["sampler_param_names"]:
                extraction[key] = []
        for key, values in zip(pyholder["sampler_param_names"], pyholder["sampler_params"]):
            extraction[key].append(values[-ndraws:])

    data = OrderedDict()
    for key, values in extraction.items():
        values = np.stack(values, axis=0)
        dtype = dtypes.get(key)
        values = values.astype(dtype)
        name = re.sub("__$", "", key)
        name = "diverging" if name == "divergent" else name
        data[name] = values

    return data


def get_draws_stan3(fit, model=None, variables=None, ignore=None):
    """Extract draws from PyStan3 fit."""
    if ignore is None:
        ignore = []

    dtypes = {}
    if model is not None:
        dtypes = infer_dtypes(fit, model)

    if variables is None:
        variables = fit.param_names
    elif isinstance(variables, str):
        variables = [variables]
    variables = list(variables)

    data = OrderedDict()

    for var in variables:
        if var in data:
            continue
        dtype = dtypes.get(var)

        # in future fix the correct number of draws if fit.save_warmup is True
        new_shape = (*fit.dims[fit.param_names.index(var)], -1, fit.num_chains)
        values = fit._draws[fit._parameter_indexes(var), :]  # pylint: disable=protected-access
        values = values.reshape(new_shape, order="F")
        values = np.moveaxis(values, [-2, -1], [1, 0])
        values = values.astype(dtype)
        data[var] = values

    return data


def get_sample_stats_stan3(fit, variables=None, ignore=None):
    """Extract sample stats from PyStan3 fit."""
    dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64}

    if isinstance(variables, str):
        variables = [variables]
    if isinstance(ignore, str):
        ignore = [ignore]

    data = OrderedDict()
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
        name = "diverging" if name == "divergent" else name
        data[name] = values

    return data


def infer_dtypes(fit, model=None):
    """Infer dtypes from Stan model code.

    Function strips out generated quantities block and searchs for `int`
    dtypes after stripping out comments inside the block.
    """
    pattern_remove_comments = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE
    )
    stan_integer = r"int"
    stan_limits = r"(?:\<[^\>]+\>)*"  # ignore group: 0 or more <....>
    stan_param = r"([^;=\s\[]+)"  # capture group: ends= ";", "=", "[" or whitespace
    stan_ws = r"\s*"  # 0 or more whitespace
    pattern_int = re.compile(
        "".join((stan_integer, stan_ws, stan_limits, stan_ws, stan_param)), re.IGNORECASE
    )
    if model is None:
        stan_code = fit.get_stancode()
        model_pars = fit.model_pars
    else:
        stan_code = model.program_code
        model_pars = fit.param_names
    # remove deprecated comments
    stan_code = "\n".join(
        line if "#" not in line else line[: line.find("#")] for line in stan_code.splitlines()
    )
    stan_code = re.sub(pattern_remove_comments, "", stan_code)
    stan_code = stan_code.split("generated quantities")[-1]
    dtypes = re.findall(pattern_int, stan_code)
    dtypes = {item.strip(): "int" for item in dtypes if item.strip() in model_pars}
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
    prior_model=None
):
    """Convert PyStan data into an InferenceData object.

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
        equal to its values is used.
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
        ).to_inference_data()
