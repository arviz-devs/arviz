"""PyStan-specific conversion code."""
from copy import deepcopy
import re

import numpy as np
import xarray as xr

from .inference_data import InferenceData
from .base import requires, dict_to_dataset, generate_dims_coords


class PyStanConverter:
    """Encapsulate PyStan specific logic."""

    def __init__(self, *_, fit=None, prior=None, posterior_predictive=None,
                 observed_data=None, log_likelihood=None, coords=None, dims=None):
        self.fit = fit
        self.prior = prior
        self.posterior_predictive = posterior_predictive
        self.observed_data = observed_data
        self.log_likelihood = log_likelihood
        self.coords = coords
        self.dims = dims
        self._var_names = fit.model_pars

    @requires('fit')
    def posterior_to_xarray(self):
        """Extract posterior samples from fit."""
        dtypes = self.infer_dtypes()
        nchain = self.fit.sim["chains"]
        data = {}
        var_dict = self.fit.extract(self._var_names, dtypes=dtypes, permuted=False)
        if not isinstance(var_dict, dict):
            # PyStan version < 2.18
            var_dict = self.fit.extract(self._var_names, dtypes=dtypes, permuted=True)
            permutation_order = self.fit.sim["permutation"]
            original_order = []
            for i_permutation_order in permutation_order:
                reorder = np.argsort(i_permutation_order) + len(original_order)
                original_order.extend(list(reorder))
            nchain = self.fit.sim["chains"]
            for key, values in var_dict.items():
                var_dict[key] = self.unpermute(values, original_order, nchain)
        # filter posterior_predictive and log_likelihood
        post_pred = self.posterior_predictive
        if post_pred is None or isinstance(post_pred, dict):
            post_pred = []
        elif isinstance(post_pred, str):
            post_pred = [post_pred]
        log_lik = self.log_likelihood
        if not isinstance(log_lik, str):
            log_lik = []
        else:
            log_lik = [log_lik]

        for var_name, values in var_dict.items():
            if var_name in post_pred+log_lik:
                continue
            if len(values.shape) == 0:
                values = np.atleast_2d(values)
            elif len(values.shape) == 1:
                if nchain == 1:
                    values = np.expand_dims(values, -1)
                else:
                    values = np.expand_dims(values, 0)
            data[var_name] = np.swapaxes(values, 0, 1)
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('fit')
    def sample_stats_to_xarray(self):
        """Extract sample_stats from fit."""
        dtypes = {
            'divergent__' : bool,
            'n_leapfrog__' : np.int64,
            'treedepth__' : np.int64,
        }

        rename_key = {
            'accept_stat__' : 'accept_stat',
            'divergent__' : 'diverging',
            'energy__' : 'energy',
            'lp__' : 'lp',
            'n_leapfrog__' : 'n_leapfrog',
            'stepsize__' : 'stepsize',
            'treedepth__' : 'treedepth',
        }

        nchain = self.fit.sim["chains"]
        sampler_params = self.fit.get_sampler_params(inc_warmup=False)
        stat_lp = self.fit.extract('lp__', permuted=False)
        log_likelihood = self.log_likelihood
        if log_likelihood is not None:
            if isinstance(log_likelihood, str):
                log_likelihood_vals = self.fit.extract(log_likelihood, permuted=False)
            else:
                log_likelihood_vals = np.asarray(log_likelihood)
        if not isinstance(stat_lp, dict):
            # PyStan version < 2.18
            permutation_order = self.fit.sim["permutation"]
            original_order = []
            for i_permutation_order in permutation_order:
                reorder = np.argsort(i_permutation_order) + len(original_order)
                original_order.extend(list(reorder))
            nchain = self.fit.sim["chains"]
            stat_lp = self.fit.extract('lp__', permuted=True)['lp__']
            stat_lp = self.unpermute(stat_lp, original_order, nchain)
            if log_likelihood is not None:
                if isinstance(log_likelihood, str):
                    log_likelihood_vals = self.fit.extract(log_likelihood, permuted=True)
                    log_likelihood_vals = log_likelihood_vals[log_likelihood]
                log_likelihood_vals = self.unpermute(log_likelihood_vals, original_order, nchain)
        else:
            # PyStan version 2.18+
            stat_lp = stat_lp['lp__']
            if len(stat_lp.shape) == 0:
                stat_lp = np.atleast_2d(stat_lp)
            elif len(stat_lp.shape) == 1:
                if nchain == 1:
                    stat_lp = np.expand_dims(stat_lp, -1)
                else:
                    stat_lp = np.expand_dims(stat_lp, 0)
            stat_lp = np.swapaxes(stat_lp, 0, 1)
            if log_likelihood is not None:
                if isinstance(log_likelihood, str):
                    log_likelihood_vals = log_likelihood_vals[log_likelihood]
                elif len(log_likelihood_vals.shape) == 1:
                    if len(log_likelihood_vals.shape) == 0:
                        log_likelihood_vals = np.atleast_2d(log_likelihood_vals)
                    elif nchain == 1:
                        log_likelihood_vals = np.expand_dims(log_likelihood, 0)
                    else:
                        log_likelihood_vals = np.expand_dims(log_likelihood, -1)
                log_likelihood_vals = np.swapaxes(log_likelihood_vals, 0, 1)
        # copy dims and coords
        dims = deepcopy(self.dims) if self.dims is not None else {}
        coords = deepcopy(self.coords) if self.coords is not None else {}

        # Add lp to sampler_params
        for i, _ in enumerate(sampler_params):
            sampler_params[i]['lp__'] = stat_lp[i]
        if log_likelihood is not None:
            # Add log_likelihood to sampler_params
            for i, _ in enumerate(sampler_params):
                # slice log_likelihood to keep dimensions
                sampler_params[i]['log_likelihood'] = log_likelihood_vals[i:i+1]
            # change dims and coords for log_likelihood if defined
            if isinstance(log_likelihood, str) and log_likelihood in dims:
                dims["log_likelihood"] = dims.pop(log_likelihood)
            if isinstance(log_likelihood, str) and log_likelihood in coords:
                coords["log_likelihood"] = coords.pop(log_likelihood)
        data = {}
        for key in sampler_params[0]:
            name = rename_key.get(key, re.sub('__$', "", key))
            data[name] = np.vstack([j[key].astype(dtypes.get(key)) for j in sampler_params])
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('fit')
    @requires('posterior_predictive')
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        nchain = self.fit.sim["chains"]
        if isinstance(self.posterior_predictive, dict):
            data = {}
            for key, values in self.posterior_predictive.items():
                if len(values.shape) == 0:
                    values = np.atleast_2d(values)
                elif len(values.shape) == 1:
                    if nchain == 1:
                        values = np.expand_dims(values, -1)
                    else:
                        values = np.expand_dims(values, 0)
                values = np.swapaxes(values, 0, 1)
                data[key] = values
        else:
            dtypes = self.infer_dtypes()
            data = {}
            var_dict = self.fit.extract(self.posterior_predictive, dtypes=dtypes, permuted=False)
            if not isinstance(var_dict, dict):
                # PyStan version < 2.18
                var_dict = self.fit.extract(self.posterior_predictive, dtypes=dtypes, permuted=True)
                permutation_order = self.fit.sim["permutation"]
                original_order = []
                for i_permutation_order in permutation_order:
                    reorder = np.argsort(i_permutation_order) + len(original_order)
                    original_order.extend(list(reorder))
                nchain = self.fit.sim["chains"]
                for key, values in var_dict.items():
                    var_dict[key] = self.unpermute(values, original_order, nchain)
            for var_name, values in var_dict.items():
                if len(values.shape) == 0:
                    values = np.atleast_2d(values)
                elif len(values.shape) == 1:
                    if nchain == 1:
                        values = np.expand_dims(values, -1)
                    else:
                        values = np.expand_dims(values, 0)
                data[var_name] = np.swapaxes(values, 0, 1)
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('fit')
    @requires('prior')
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        nchain = self.fit.sim["chains"]
        data = {}
        for key, values in self.prior.items():
            if len(values.shape) == 0:
                values = np.atleast_2d(values)
            elif len(values.shape) == 1:
                if nchain == 1:
                    values = np.expand_dims(values, -1)
                else:
                    values = np.expand_dims(values, 0)
            values = np.swapaxes(values, 0, 1)
            data[key] = values
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('fit')
    @requires('observed_data')
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
        if isinstance(self.observed_data, dict):
            observed_data = {}
            for key, vals in self.observed_data.items():
                vals = np.atleast_1d(vals)
                val_dims = dims.get(key)
                val_dims, coords = generate_dims_coords(vals.shape, key,
                                                        dims=val_dims, coords=self.coords)
                observed_data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        else:
            if isinstance(self.observed_data, str):
                observed_names = [self.observed_data]
            else:
                observed_names = self.observed_data
            observed_data = {}
            for key in observed_names:
                vals = np.atleast_1d(self.fit.data[key])
                val_dims = dims.get(key)
                val_dims, coords = generate_dims_coords(vals.shape, key,
                                                        dims=val_dims, coords=self.coords)
                observed_data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=observed_data)

    @requires('fit')
    def infer_dtypes(self):
        """Infer dtypes from Stan model code.

        Function strips out generated quantities block and searchs for `int`
        dtypes after stripping out comments inside the block.
        """
        pattern_remove_comments = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL|re.MULTILINE
        )
        stan_integer = r"int"
        stan_limits = r"(?:\<[^\>]+\>)*" # ignore group: 0 or more <....>
        stan_param = r"([^;=\s\[]+)" # capture group: ends= ";", "=", "[" or whitespace
        stan_ws = r"\s*" # 0 or more whitespace
        pattern_int = re.compile(
            "".join((stan_integer, stan_ws, stan_limits, stan_ws, stan_param)),
            re.IGNORECASE
        )
        stan_code = self.fit.get_stancode()
        # remove deprecated comments
        stan_code = "\n".join(\
                line if "#" not in line else line[:line.find("#")]\
                for line in stan_code.splitlines())
        stan_code = re.sub(pattern_remove_comments, "", stan_code)
        stan_code = stan_code.split("generated quantities")[-1]
        dtypes = re.findall(pattern_int, stan_code)
        dtypes = {item.strip() : 'int' for item in dtypes if item.strip() in self._var_names}
        return dtypes

    def unpermute(self, ary, idx, nchain):
        """Unpermute permuted sample.

        Returns output compatible with PyStan 2.18+
        fit.extract(par, permuted=False)[par]

        Parameters
        ----------
        ary : list
            Permuted sample
        idx : list
            list containing reorder indexes.
            `idx = np.argsort(permutation_order)`
        nchain : int
            number of chains used
            `fit.sim['chains']`

        Returns
        -------
        numpy.ndarray
            Unpermuted sample
        """
        ary = np.asarray(ary)[idx]
        if ary.shape:
            ary_shape = ary.shape[1:]
        else:
            ary_shape = ary.shape
        ary = ary.reshape((-1, nchain, *ary_shape), order='F')
        return ary

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `trace`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        return InferenceData(**{
            'posterior': self.posterior_to_xarray(),
            'sample_stats': self.sample_stats_to_xarray(),
            'posterior_predictive' : self.posterior_predictive_to_xarray(),
            'prior' : self.prior_to_xarray(),
            'observed_data' : self.observed_data_to_xarray(),
        })


def from_pystan(*, fit=None, prior=None, posterior_predictive=None,
                observed_data=None, log_likelihood=None, coords=None, dims=None):
    """Convert pystan data into an InferenceData object.

    Parameters
    ----------
    fit : StanFit4Model
        PyStan fit object.
    prior : dict
        A dictionary containing prior samples extracted from pystan fit object.
        For PyStan 2.18+:
            `prior_dict = prior_fit.extract(pars=prior_vars, permuted=False)`
        For PyStan 2.17 and earlier:
            `prior_dict = prior_fit.extract(pars=prior_vars)`
            `prior_dict = {k : az.from_pystan.unpermute(v) for k, v in prior_dict.items()}`
    posterior_predictive : str, a list of str or dict
        Posterior predictive samples for the fit. If given string or a list of strings
        function extracts values from the fit object. Else a dictionary of posterior samples
        is assumed in PyStan extract format.
        For PyStan 2.18+:
            `pp_dict = posterior_predictive_fit.extract(pars=pp_vars, permuted=False)`
        For PyStan 2.17 and earlier:
            `pp_dict = posterior_predictive_fit.extract(pars=prior_vars)`
            `pp_dict = {k : az.from_pystan.unpermute(v) for k, v in pp_dict.items()}`
    observed_data : str or a list of str or a dictionary
        observed data used in the sampling. If a str or a list of str is given, observed data is
         extracted from the `fit.data`. Else a dictionary is assumed containing observed data.
    log_likelihood : str or np.ndarray
        log_likelihood for data calculated elementwise. If a string is given, log_likelihood is
        extracted from the fit object. Else a ndarray containing elementwise log_likelihood is
        assumed in PyStan extract format.
        For PyStan 2.18+:
            `log_likelihood = log_likelihood_fit.extract(pars=log_likelihood_var, permuted=False)`
            `log_likelihood = log_likelihood[log_likelihood_var]`
        For PyStan 2.17 and earlier:
            `log_likelihood = log_likelihood_fit.extract(pars=log_likelihood_var)`
            `log_likelihood = az.from_pystan.unpermute(log_likelihood[log_likelihood_var])`
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable.

    Returns
    -------
    InferenceData object
    """
    return PyStanConverter(
        fit=fit,
        prior=prior,
        posterior_predictive=posterior_predictive,
        observed_data=observed_data,
        log_likelihood=log_likelihood,
        coords=coords,
        dims=dims).to_inference_data()
