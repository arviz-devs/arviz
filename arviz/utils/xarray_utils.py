"""Utilities for converting and working with netcdf and xarray data."""
import re
import warnings
from copy import deepcopy

import numpy as np
import xarray as xr

from ..inference_data import InferenceData
from ..compat import pymc3 as pm


def convert_to_inference_data(obj, *, group='posterior', coords=None, dims=None):
    """Convert a supported object to an InferenceData object.

    This function sends `obj` to the right conversion function. It is idempotent,
    in that it will return arviz.InferenceData objects unchanged.

    Parameters
    ----------
    obj : dict, str, np.ndarray, xr.Dataset, pystan fit, pymc3 trace
        A supported object to convert to InferenceData:
            InferenceData: returns unchanged
            str: Attempts to load the netcdf dataset from disk
            pystan fit: Automatically extracts data
            pymc3 trace: Automatically extracts data
            xarray.Dataset: adds to InferenceData as only group
            dict: creates an xarray dataset as the only group
            numpy array: creates an xarray dataset as the only group, gives the
                         array an arbitrary name
    group : str
        If `obj` is a dict or numpy array, assigns the resulting xarray
        dataset to this group. Default: "posterior".
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable

    Returns
    -------
    InferenceData
    """
    # Cases that convert to InferenceData
    if isinstance(obj, InferenceData):
        return obj
    elif isinstance(obj, str):
        return InferenceData.from_netcdf(obj)
    elif obj.__class__.__name__ == 'StanFit4Model':  # ugly, but doesn't make PyStan a requirement
        return pystan_to_inference_data(fit=obj, coords=coords, dims=dims)
    elif obj.__class__.__name__ == 'MultiTrace':  # ugly, but doesn't make PyMC3 a requirement
        return pymc3_to_inference_data(trace=obj, coords=coords, dims=dims)

    # Cases that convert to xarray
    if isinstance(obj, xr.Dataset):
        dataset = obj
    elif isinstance(obj, dict):
        dataset = dict_to_dataset(obj, coords=coords, dims=dims)
    elif isinstance(obj, np.ndarray):
        dataset = dict_to_dataset({'x': obj}, coords=coords, dims=dims)
    else:
        allowable_types = (
            'xarray dataset',
            'dict',
            'netcdf file',
            'numpy array',
            'pystan fit',
            'pymc3 trace'
        )
        raise ValueError('Can only convert {} to InferenceData, not {}'.format(
            ', '.join(allowable_types), obj.__class__.__name__))

    return InferenceData(**{group: dataset})


def convert_to_dataset(obj, *, group='posterior', coords=None, dims=None):
    """Convert a supported object to an xarray dataset.

    This function is idempotent, in that it will return xarray.Dataset functions
    unchanged. Raises `ValueError` if the desired group can not be extracted.

    Note this goes through a DataInference object. See `convert_to_inference_data`
    for more details. Raises ValueError if it can not work out the desired
    conversion.

    Parameters
    ----------
    obj : dict, str, np.ndarray, xr.Dataset, pystan fit, pymc3 trace
        A supported object to convert to InferenceData:
            InferenceData: returns unchanged
            str: Attempts to load the netcdf dataset from disk
            pystan fit: Automatically extracts data
            pymc3 trace: Automatically extracts data
            xarray.Dataset: adds to InferenceData as only group
            dict: creates an xarray dataset as the only group
            numpy array: creates an xarray dataset as the only group, gives the
                         array an arbitrary name
    group : str
        If `obj` is a dict or numpy array, assigns the resulting xarray
        dataset to this group.
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable

    Returns
    -------
    xarray.Dataset
    """
    inference_data = convert_to_inference_data(obj, group=group, coords=coords, dims=dims)
    dataset = getattr(inference_data, group, None)
    if dataset is None:
        raise ValueError('Can not extract {group} from {obj}! See {filename} for other '
                         'conversion utilities.'.format(group=group, obj=obj, filename=__file__))
    return dataset


class requires: # pylint: disable=invalid-name
    """Decorator to return None if an object does not have the required attribute."""

    def __init__(self, *props):
        self.props = props

    def __call__(self, func):
        """Wrap the decorated function."""
        def wrapped(cls, *args, **kwargs):
            """Return None if not all props are available."""
            for prop in self.props:
                if getattr(cls, prop) is None:
                    return None
            return func(cls, *args, **kwargs)
        return wrapped

def _generate_dims_coords(shape, var_name, dims=None, coords=None, default_dims=None):
    """Generate default dimensions and coordinates for a variable.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the variable
    var_name : str
        Name of the variable. Used in the default name, if necessary
    dims : list
        List of dimensions for the variable
    coords : dict[str] -> list[str]
        Map of dimensions to coordinates
    default_dims : list[str]
        Dimensions that do not apply to the variable's shape

    Returns
    -------
    list[str]
        Default dims
    dict[str] -> list[str]
        Default coords
    """
    if default_dims is None:
        default_dims = []
    if dims is None:
        dims = []
    if len([dim for dim in dims if dim not in default_dims]) > len(shape):
        warnings.warn('More dims ({dims_len}) given than exists ({shape_len}). '
                      'Passed array should have shape (chains, draws, *shape)'.format(
                          dims_len=len(dims), shape_len=len(shape)), SyntaxWarning)
    if coords is None:
        coords = {}

    coords = deepcopy(coords)
    dims = deepcopy(dims)

    for idx, dim_len in enumerate(shape):
        if (len(dims) < idx+1) or (dims[idx] is None):
            dim_name = '{var_name}_dim_{idx}'.format(var_name=var_name, idx=idx)
            if len(dims) < idx + 1:
                dims.append(dim_name)
            else:
                dims[idx] = dim_name
        dim_name = dims[idx]
        if dim_name not in coords:
            coords[dim_name] = np.arange(dim_len)
    return dims, coords


def numpy_to_data_array(ary, *, var_name='data', coords=None, dims=None):
    """Convert a numpy array to an xarray.DataArray.

    The first two dimensions will be (chain, draw), and any remaining
    dimensions will be "shape".
    If the numpy array is 1d, this dimension is interpreted as draw
    If the numpy array is 2d, it is interpreted as (chain, draw)
    If the numpy array is 3 or more dimensions, the last dimensions are kept as shapes.

    Parameters
    ----------
    ary : np.ndarray
        A numpy array. If it has 2 or more dimensions, the first dimension should be
        independent chains from a simulation. Use `np.expand_dims(ary, 0)` to add a
        single dimension to the front if there is only 1 chain.
    var_name : str
        If there are no dims passed, this string is used to name dimensions
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : List(str)
        A list of coordinate names for the variable

    Returns
    -------
    xr.DataArray
        Will have the same data as passed, but with coordinates and dimensions
    """
    # manage and transform copies
    default_dims = ["chain", "draw"]
    ary = np.atleast_2d(ary)
    n_chains, n_samples, *shape = ary.shape
    if n_chains > n_samples:
        warnings.warn('More chains ({n_chains}) than draws ({n_samples}). '
                      'Passed array should have shape (chains, draws, *shape)'.format(
                          n_chains=n_chains, n_samples=n_samples), SyntaxWarning)

    dims, coords = _generate_dims_coords(shape, var_name,
                                         dims=dims,
                                         coords=coords,
                                         default_dims=default_dims)

    # reversed order for default dims: 'chain', 'draw'
    if 'draw' not in dims:
        dims = ['draw'] + dims
    if 'chain' not in dims:
        dims = ['chain'] + dims

    if 'chain' not in coords:
        coords['chain'] = np.arange(n_chains)
    if 'draw' not in coords:
        coords['draw'] = np.arange(n_samples)

    # filter coords based on the dims
    coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in dims}
    return xr.DataArray(ary, coords=coords, dims=dims)


def dict_to_dataset(data, *, coords=None, dims=None):
    """Convert a dictionary of numpy arrays to an xarray.Dataset.

    Examples
    --------
    dict_to_dataset({'x': np.random.randn(4, 100), 'y', np.random.rand(4, 100)})
    """
    if dims is None:
        dims = {}

    data_vars = {}
    for key, values in data.items():
        data_vars[key] = numpy_to_data_array(values,
                                             var_name=key,
                                             coords=coords,
                                             dims=dims.get(key))
    return xr.Dataset(data_vars=data_vars)


class PyMC3Converter:
    """Encapsulate PyMC3 specific logic."""

    def __init__(self, *_, trace=None, prior=None, posterior_predictive=None,
                 coords=None, dims=None):
        self.trace = trace
        self.prior = prior
        self.posterior_predictive = posterior_predictive
        self.coords = coords
        self.dims = dims

    @requires('trace')
    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        var_names = pm.utils.get_default_varnames(self.trace.varnames, include_transformed=False)
        data = {}
        for var_name in var_names:
            data[var_name] = np.array(self.trace.get_values(var_name, combine=False))
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('trace')
    def sample_stats_to_xarray(self):
        """Extract sample_stats from PyMC3 trace."""
        data = {}
        for stat in self.trace.stat_names:
            data[stat] = np.array(self.trace.get_sampler_stats(stat, combine=False))
        return dict_to_dataset(data)

    @requires('posterior_predictive')
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        data = {k: np.expand_dims(v, 0) for k, v in self.posterior_predictive.items()}
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('prior')
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        return dict_to_dataset({k: np.expand_dims(v, 0) for k, v in self.prior.items()},
                               coords=self.coords,
                               dims=self.dims)

    @requires('trace')
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        # This next line is brittle and may not work forever, but is a secret
        # way to access the model from the trace.
        model = self.trace._straces[0].model # pylint: disable=protected-access

        observations = {obs.name: obs.observations for obs in model.observed_RVs}
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
        observed_data = {}
        for name, vals in observations.items():
            vals = np.atleast_1d(vals)
            val_dims = dims.get(name)
            val_dims, coords = _generate_dims_coords(vals.shape, name,
                                                     dims=val_dims, coords=self.coords)
            # filter coords based on the dims
            coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in val_dims}
            observed_data[name] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=observed_data)


    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `trace`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        return InferenceData(**{
            'posterior': self.posterior_to_xarray(),
            'sample_stats': self.sample_stats_to_xarray(),
            'posterior_predictive': self.posterior_predictive_to_xarray(),
            'prior': self.prior_to_xarray(),
            'observed_data': self.observed_data_to_xarray(),
        })


class PyStanConverter:
    """Encapsulate PyStan specific logic."""

    def __init__(self, *_, fit=None, prior=None, posterior_predictive=None,
                 observed_data=None, coords=None, dims=None):
        self.fit = fit
        self.prior = prior
        self.posterior_predictive = posterior_predictive
        self.observed_data = observed_data
        self.coords = coords
        self.dims = dims
        self._var_names = fit.model_pars

    @requires('fit')
    def posterior_to_xarray(self):
        """Extract posterior samples from fit."""
        dtypes = self.infer_dtypes()
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
        post_pred = self.posterior_predictive
        if post_pred is None or isinstance(post_pred, dict):
            post_pred = []
        elif isinstance(post_pred, str):
            post_pred = [post_pred]
        for var_name, values in var_dict.items():
            if var_name in post_pred:
                continue
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

        sampler_params = self.fit.get_sampler_params(inc_warmup=False)
        stat_lp = self.fit.extract('lp__', permuted=False)
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
        else:
            # PyStan version 2.18+
            stat_lp = stat_lp['lp__']
        # Add lp to sampler_params
        for i, _ in enumerate(sampler_params):
            sampler_params[i]['lp__'] = stat_lp[:, i]
        data = {}
        for key in sampler_params[0]:
            name = rename_key.get(key, re.sub('__$', "", key))
            data[name] = np.vstack([j[key].astype(dtypes.get(key)) for j in sampler_params])
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('fit')
    @requires('posterior_predictive')
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        if isinstance(self.posterior_predictive, dict):
            data = {k : np.swapaxes(v, 0, 1)
                    for k, v in self.posterior_predictive.items()}
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
                data[var_name] = np.swapaxes(values, 0, 1)
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('prior')
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        data = {k : np.swapaxes(v, 0, 1)
                for k, v in self.prior.items()}
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('fit')
    @requires('observed_data')
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        if isinstance(self.observed_data, str):
            observed_names = [self.observed_data]
        else:
            observed_names = self.observed_data
        observed_data = {}
        for key in observed_names:
            vals = np.atleast_1d(self.fit.data[key])
            val_dims, coords = _generate_dims_coords(vals.shape, key,
                                                     dims=None, coords=self.coords)
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
            `fit.sim['chains']´

        Returns
        -------
        numpy.ndarray
            Unpermuted sample
        """
        ary = np.asarray(ary)[idx]
        ary_shape = ary.shape[1:]
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


def pymc3_to_inference_data(*, trace=None, prior=None, posterior_predictive=None,
                            coords=None, dims=None):
    """Convert pymc3 data into an InferenceData object."""
    return PyMC3Converter(
        trace=trace,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords=coords,
        dims=dims).to_inference_data()


def pystan_to_inference_data(*, fit=None, prior=None, posterior_predictive=None,
                             observed_data=None, coords=None, dims=None):
    """Convert pystan data into an InferenceData object."""
    return PyStanConverter(
        fit=fit,
        prior=prior,
        posterior_predictive=posterior_predictive,
        observed_data=observed_data,
        coords=coords,
        dims=dims).to_inference_data()
