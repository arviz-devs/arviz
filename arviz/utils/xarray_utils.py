from abc import ABC, abstractmethod, abstractstaticmethod
from copy import deepcopy as copy
import os
import re
import tempfile

import numpy as np
import xarray as xr

from arviz import InferenceData, config
from arviz.compat import pymc3 as pm


def get_converter(obj, *_, filename=None, coords=None, dims=None, chains=None):
    """Get the converter to transform a supported object to an xarray dataset.

    This function sends `obj` to the right conversion function. It is idempotent,
    in that it will return arviz.InferenceData objects unchanged.

    Parameters
    ----------
    obj : A dict, or an object from PyStan or PyMC3 to convert
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, Tuple(str)]
        A mapping from pymc3 variables to a tuple corresponding to
        the shape of the variable, where the elements of the tuples are
        the names of the coordinate dimensions.
    chains : int or None
        The number of chains sampled from the posterior, only necessary for
        converting dicts.

    Returns
    -------
    InferenceData
        The coordinates are those passed in and ('chain', 'draw')
    """
    if isinstance(obj, dict):
        return DictToNetCDF(obj, filename=filename, coords=coords, dims=dims, chains=chains)
    elif obj.__class__.__name__ == 'StanFit4Model':  # ugly, but doesn't make PyStan a requirement
        return PyStanToNetCDF(obj, filename=filename, coords=coords, dims=dims)
    elif obj.__class__.__name__ == 'MultiTrace':  # ugly, but doesn't make PyMC3 a requirement
        return PyMC3ToNetCDF(obj, filename=filename, coords=coords, dims=dims)
    else:
        raise TypeError('Can only convert PyStan or PyMC3 object to xarray, not {}'.format(
            obj.__class__.__name__))


def convert_to_netcdf(obj, *_, filename=None, coords=None, dims=None, chains=None):
    """Convert a supported object to a netCDF dataset

    This function sends `obj` to the right conversion function. It is idempotent,
    in that it will return `arviz.InferenceData` unchanged.

    Parameters
    ----------
    obj : A dict, or an object from PyStan or PyMC3 to convert
    filename : str
        Location to store the netCDF dataset
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, Tuple(str)]
        A mapping from pymc3 variables to a tuple corresponding to
        the shape of the variable, where the elements of the tuples are
        the names of the coordinate dimensions.
    chains : int or None
        The number of chains sampled from the posterior, only necessary for
        converting dicts.

    Returns
    -------
    InferenceData
        This wraps a netCDF datset representing those groups available to the object.
        The coordinates are those passed in and ('chain', 'draw')
    """
    if isinstance(obj, InferenceData):
        return obj
    else:
        return get_converter(obj,
                             filename=filename,
                             coords=coords,
                             dims=dims,
                             chains=chains).to_netcdf()


class Converter(ABC):
    def __init__(self, obj, filename=None, coords=None, dims=None):
        self.obj = obj
        if filename is None:
            directory = config['default_data_directory']
            if not os.path.exists(directory):
                os.mkdir(directory)
            _, filename = tempfile.mkstemp(prefix='arviz_', dir=directory, suffix='.nc')
        self.filename = filename
        # pylint: disable=assignment-from-none
        self.varnames, self.coords, self.dims = self.default_varnames_coords_dims(obj, coords, dims)
        self.verified, self.warning = self.verify_coords_dims()
        self.converters = {
            'posterior': 'posterior_to_xarray',
            'sample_stats': 'sample_stats_to_xarray',
        }

    def to_netcdf(self):
        has_group = False
        mode = 'w' # overwrite first, then append
        for group, func_name in self.converters.items():
            if hasattr(self, func_name):
                data = getattr(self, func_name)()
                try:
                    data.to_netcdf(self.filename, mode=mode, group=group)
                except PermissionError as err:
                    msg = 'File "{}" is in use - is another object using it?'.format(self.filename)
                    raise PermissionError(msg) from err
                has_group = True
                mode = 'a'
        if not has_group:
            msg = ('{} has no functions creating groups! Must implement one of '
                   'the following functions:\n{}'.format(
                       self.__class__.__name__, '\n'.join(self.converters.values())))
            raise RuntimeError(msg)
        return InferenceData(self.filename)

    @abstractstaticmethod
    def default_varnames_coords_dims(obj, coords, dims):  # pylint: disable=unused-argument
        return

    @abstractmethod
    def verify_coords_dims(self):
        return


class DictToNetCDF(Converter):
    def __init__(self, obj, filename=None, coords=None, dims=None, chains=None):
        """Convert a dict containing posterior samples to an InferenceData object.

        Parameters
        ----------
        obj : dict[str, iterable]
            The values should have shape (draws, chains, *rv_shape),
            where rv_shape is the shape of the random variable.
        coords : dict[str, iterable]
            A dictionary containing the values that are used as index. The key
            is the name of the dimension, the values are the index values.
        dims : dict[str, Tuple(str)]
            A mapping from pystan variables to a tuple corresponding to
            the shape of the variable, where the elements of the tuples are
            the names of the coordinate dimensions.
        chains : int
            Number of chains in the numpy array. Defaults to 1.

        Returns
        -------
        InferenceData
            The coordinates are those passed in and ('chain', 'draw')
        """
        if chains is None:
            raise ValueError('Number of chains required when converting a dict')

        if coords is None:
            coords = {}

        coords['chain'] = np.arange(chains)
        super().__init__(obj, filename=filename, coords=coords, dims=dims)

    def posterior_to_xarray(self):
        """Extract posterior data from a dictionary"""
        data = xr.Dataset(coords=self.coords)
        base_dims = ['chain', 'draw']

        for key in self.varnames:
            vals = self.obj[key]
            if len(vals.shape) == 1:
                vals = np.expand_dims(vals, axis=1)
            vals = np.swapaxes(vals, 0, 1)
            dims_str = base_dims + self.dims[key]
            try:
                coords = {v: self.coords[v] for v in dims_str if v in self.coords}
                data[key] = xr.DataArray(vals, coords=coords, dims=dims_str)
            except (KeyError, ValueError) as exc:
                if not self.verified:
                    raise TypeError(self.warning) from exc
                else:
                    raise exc
        return data

    @staticmethod
    def default_varnames_coords_dims(obj, coords, dims):
        """Set up varnames, coordinates, and dimensions for .posterior_to_xarray function

        obj : dict[str, iterable]
        coords : dict[str, iterable]
            A dictionary containing the values that are used as index. The key
            is the name of the dimension, the values are the index values.
        dims : dict[str, Tuple(str)]
            A mapping from pymc3 variables to a tuple corresponding to
            the shape of the variable, where the elements of the tuples are
            the names of the coordinate dimensions.

        Returns
        -------
        iterable[str]
            The non-transformed variable names from the trace
        dict[str, iterable]
            Default coordinates for the trace
        dict[str, Tuple(str)]
            Default dimensions for the xarray
        """
        varnames = list(obj.keys())

        coords['draw'] = np.arange(obj[varnames[0]].shape[0])
        coords = {key: xr.IndexVariable((key,), data=vals) for key, vals in coords.items()}

        if dims is None:
            dims = {}

        for varname in varnames:
            if varname not in dims:
                vals = obj[varname]
                if len(vals.shape) == 1:
                    vals = np.expand_dims(vals, axis=1)
                vals = np.swapaxes(vals, 0, 1)
                shape_len = len(vals.shape)
                if shape_len == 2:
                    dims[varname] = []
                else:
                    dims[varname] = [
                        "{}_dim_{}".format(varname, idx)
                        for idx in range(1, shape_len-2+1)
                    ]

        return varnames, coords, dims

    def verify_coords_dims(self):
        """Light checking and guessing on the structure of an xarray

        Returns
        -------
        bool
            Whether it passes the check
        str
            Warning string in case it does not pass
        """
        inferred_coords = copy(self.coords)
        inferred_dims = copy(self.dims)
        for key in ('draw', 'chain'):
            inferred_coords.pop(key)
        global_coords = {}
        throw = False

        for varname in self.varnames:
            vals = self.obj[varname]
            if len(vals.shape) == 1:
                vals = np.expand_dims(vals, axis=1)
            vals = np.swapaxes(vals, 0, 1)
            shapes = [d for shape in self.coords.values() for d in shape.shape]
            for idx, shape in enumerate(vals[0].shape[1:], 1):
                try:
                    shapes.remove(shape)
                except ValueError:
                    throw = True
                    if shape not in global_coords:
                        global_coords[shape] = '{}_dim_{}'.format(varname, idx)
                    key = global_coords[shape]
                    inferred_dims[varname].append(key)
                    if key not in inferred_coords:
                        inferred_coords[key] = 'np.arange({})'.format(shape)
        if throw:
            inferred_dims = {k: v for k, v in inferred_dims.items() if v}
            msg = 'Bad arguments! Try setting\ncoords={}\ndims={}'.format(
                inferred_coords, inferred_dims
            )
            return False, msg
        return True, ''


class PyMC3ToNetCDF(Converter):
    def __init__(self, trace, filename=None, coords=None, dims=None):
        """Convert a pymc3 trace to an InferenceData object

        Parameters
        ----------
        trace : pymc3 trace
        coords : dict[str, iterable]
            A dictionary containing the values that are used as index. The key
            is the name of the dimension, the values are the index values.
        dims : dict[str, Tuple(str)]
            A mapping from pymc3 variables to a tuple corresponding to
            the shape of the variable, where the elements of the tuples are
            the names of the coordinate dimensions.

        Returns
        -------
        InferenceData
            The coordinates are those passed in and ('chain', 'draw')
        """
        super().__init__(trace, filename=filename, coords=coords, dims=dims)

    def sample_stats_to_xarray(self):
        """Extract sampler statistics from PyMC3 trace."""
        dims = ['chain', 'draw']
        coords = {d: self.coords[d] for d in dims}

        sampler_stats = xr.Dataset(coords=coords)
        for key in sorted(self.obj.stat_names):
            vals = np.array(self.obj.get_sampler_stats(key, combine=False, squeeze=False))
            sampler_stats[key] = xr.DataArray(vals, coords=coords, dims=dims)
        return sampler_stats

    def posterior_to_xarray(self):
        """Extract posterior from PyMC3 trace."""
        varnames, coords, dims = self.varnames, self.coords, self.dims

        data = xr.Dataset(coords=coords)
        base_dims = ['chain', 'draw']
        for key in varnames:
            vals = self.obj.get_values(key, combine=False, squeeze=False)
            vals = np.array(vals)
            dims_str = base_dims + dims[key]
            try:
                data[key] = xr.DataArray(vals,
                                         coords={v: coords[v] for v in dims_str if v in coords},
                                         dims=dims_str)
            except KeyError as exc:
                if not self.verified:
                    raise TypeError(self.warning) from exc
                else:
                    raise exc

        return data

    @staticmethod
    def default_varnames_coords_dims(obj, coords, dims):
        """Set up varnames, coordinates, and dimensions for .posterior_to_xarray function

        obj : pymc3 trace
        coords : dict[str, iterable]
            A dictionary containing the values that are used as index. The key
            is the name of the dimension, the values are the index values.
        dims : dict[str, Tuple(str)]
            A mapping from pymc3 variables to a tuple corresponding to
            the shape of the variable, where the elements of the tuples are
            the names of the coordinate dimensions.

        Returns
        -------
        iterable[str]
            The non-transformed variable names from the trace
        dict[str, iterable]
            Default coordinates for the trace
        dict[str, Tuple(str)]
            Default dimensions for the xarray
        """
        varnames = pm.utils.get_default_varnames(obj.varnames, include_transformed=False)
        if coords is None:
            coords = {}

        coords['draw'] = np.arange(len(obj))
        coords['chain'] = np.arange(obj.nchains)
        coords = {key: xr.IndexVariable((key,), data=vals) for key, vals in coords.items()}

        if dims is None:
            dims = {}

        for varname in varnames:
            if varname not in dims:
                vals = obj.get_values(varname, combine=False, squeeze=False)
                vals = np.array(vals)
                shape_len = len(vals.shape)
                if shape_len == 2:
                    dims[varname] = []
                else:
                    dims[varname] = [
                        "{}_dim_{}".format(varname, idx)
                        for idx in range(1, shape_len-2+1)
                    ]

        return varnames, coords, dims

    def verify_coords_dims(self):
        """Light checking and guessing on the structure of an xarray for a PyMC3 trace

        Returns
        -------
        bool
            Whether it passes the check
        str
            Warning string in case it does not pass
        """
        inferred_coords = copy(self.coords)
        inferred_dims = copy(self.dims)
        for key in ('draw', 'chain'):
            inferred_coords.pop(key)
        global_coords = {}
        throw = False

        for varname in self.varnames:
            vals = self.obj.get_values(varname, combine=False, squeeze=False)
            shapes = [d for shape in self.coords.values() for d in shape.shape]
            for idx, shape in enumerate(vals[0].shape[1:], 1):
                try:
                    shapes.remove(shape)
                except ValueError:
                    throw = True
                    if shape not in global_coords:
                        global_coords[shape] = '{}_dim_{}'.format(varname, idx)
                    key = global_coords[shape]
                    inferred_dims[varname].append(key)
                    if key not in inferred_coords:
                        inferred_coords[key] = 'np.arange({})'.format(shape)
        if throw:
            inferred_dims = {k: v for k, v in inferred_dims.items() if v}
            msg = 'Bad arguments! Try setting\ncoords={}\ndims={}'.format(
                inferred_coords, inferred_dims
            )
            return False, msg
        return True, ''


class PyStanToNetCDF(Converter):
    def __init__(self, fit, filename=None, coords=None, dims=None):
        """Convert a PyStan StanFit4Model-object to an InferenceData object.

        Parameters
        ----------
        fit : StanFit4Model
        coords : dict[str, iterable]
            A dictionary containing the values that are used as index. The key
            is the name of the dimension, the values are the index values.
        dims : dict[str, Tuple(str)]
            A mapping from pystan variables to a tuple corresponding to
            the shape of the variable, where the elements of the tuples are
            the names of the coordinate dimensions.

        Returns
        -------
        InferenceData
            The coordinates are those passed in and ('chain', 'draw')
        """
        super().__init__(fit, filename=filename, coords=coords, dims=dims)

    def sample_stats_to_xarray(self):
        """Extract sampler statistics from PyStan fit."""
        dims = ['chain', 'draw']
        coords = {d: self.coords[d] for d in dims}

        sample_stats_list = self.obj.get_sampler_params(inc_warmup=False)
        sample_stats_keys = list(sample_stats_list[0].keys()) + ['lp__']
        sample_stats_keys = sorted(sample_stats_keys)

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

        sampler_stats = xr.Dataset(coords=coords)
        for key in sorted(sample_stats_keys):
            if key != 'lp__':
                vals_list = []
                for sample_stats_dict in sample_stats_list:
                    vals = sample_stats_dict[key]
                    vals = vals.astype(dtypes.get(key, np.float64))
                    vals_list.append(vals)
                vals = np.vstack(vals_list)
            else:
                vals = self.obj.extract(pars=['lp__'], permuted=False)
                vals = vals['lp__']
                vals = vals.T
            # remove __ from end
            key = rename_key.get(key, re.sub('__$', "", key))
            sampler_stats[key] = xr.DataArray(vals, coords=coords, dims=dims)
        return sampler_stats

    def posterior_to_xarray(self):
        """Extract posterior data from a pystan fit"""
        fit = self.obj
        dtypes = self.infer_dtypes()

        data = xr.Dataset(coords=self.coords)
        base_dims = ['chain', 'draw']
        extract = fit.extract(pars=self.varnames,
                              dtypes=dtypes,
                              permuted=False)
        for key, vals in extract.items():
            if len(vals.shape) == 1:
                vals = np.expand_dims(vals, axis=1)
            vals = np.swapaxes(vals, 0, 1)
            dims_str = base_dims + self.dims[key]
            try:
                coords = {v: self.coords[v] for v in dims_str if v in self.coords}
                data[key] = xr.DataArray(vals, coords=coords, dims=dims_str)
            except (KeyError, ValueError) as exc:
                if not self.verified:
                    raise TypeError(self.warning) from exc
                else:
                    raise exc

        return data

    @staticmethod
    def default_varnames_coords_dims(obj, coords, dims):
        """Set up varnames, coordinates, and dimensions for .posterior_to_xarray function

        obj : StanFit4Model
        coords : dict[str, iterable]
            A dictionary containing the values that are used as index. The key
            is the name of the dimension, the values are the index values.
        dims : dict[str, Tuple(str)]
            A mapping from pymc3 variables to a tuple corresponding to
            the shape of the variable, where the elements of the tuples are
            the names of the coordinate dimensions.

        Returns
        -------
        iterable[str]
            The non-transformed variable names from the trace
        dict[str, iterable]
            Default coordinates for the trace
        dict[str, Tuple(str)]
            Default dimensions for the xarray
        """
        varnames = obj.model_pars
        if coords is None:
            coords = {}

        coords['draw'] = np.arange(obj.sim['n_save'][0] - obj.sim['warmup2'][0])
        coords['chain'] = np.arange(obj.sim['chains'])
        coords = {key: xr.IndexVariable((key,), data=vals) for key, vals in coords.items()}

        if dims is None:
            dims = {}
        for varname, dim in zip(obj.sim['pars_oi'], obj.sim['dims_oi']):
            if varname not in dims:
                if varname == 'lp__':
                    continue
                dims[varname] = [
                    "{}_dim_{}".format(varname, idx)
                    for idx in range(1, len(dim)+1)
                ]

        return varnames, coords, dims

    def verify_coords_dims(self):
        """Light checking and guessing on the structure of an xarray for a PyStan fit

        Returns
        -------
        bool
            Whether it passes the check
        str
            Warning string in case it does not pass
        """
        fit = self.obj
        if fit.mode == 1:
            msg = "Stan model '{}' is of mode 'test_grad';\n"\
                  "sampling is not conducted.".format(fit.model_name)
            return False, msg
        elif fit.mode == 2:
            msg = "Stan model '{}' does not contain samples.".format(fit.model_name)
            return False, msg

        inferred_coords = copy(self.coords)
        inferred_dims = copy(self.dims)
        for key in ('draw', 'chain'):
            inferred_coords.pop(key)
        global_coords = {}
        throw = False

        #infer dtypes
        dtypes = self.infer_dtypes()

        for varname in self.varnames:
            var_dtype = {varname : 'int'} if varname in dtypes else {}
            # no support for pystan <= 2.17.1
            vals = fit.extract(varname, dtypes=var_dtype, permuted=False)[varname]
            if len(vals.shape) == 1:
                vals = np.expand_dims(vals, axis=1)
            vals = np.swapaxes(vals, 0, 1)
            shapes = [d for shape in self.coords.values() for d in shape.shape]
            for idx, shape in enumerate(vals[0].shape[1:], 1):
                try:
                    shapes.remove(shape)
                except ValueError:
                    throw = True
                    if shape not in global_coords:
                        global_coords[shape] = '{}_dim_{}'.format(varname, idx)
                    key = global_coords[shape]
                    inferred_dims[varname].append(key)
                    if key not in inferred_coords:
                        inferred_coords[key] = 'np.arange({})'.format(shape)
        if throw:
            inferred_dims = {k: v for k, v in inferred_dims.items() if v}
            msg = 'Bad arguments! Try setting\ncoords={}\ndims={}'.format(
                inferred_coords, inferred_dims
            )
            return False, msg
        return True, ''

    def infer_dtypes(self):
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
        stan_code = self.obj.get_stancode()
        # remove deprecated comments
        stan_code = "\n".join(\
                line if "#" not in line else line[:line.find("#")]\
                for line in stan_code.splitlines())
        stan_code = re.sub(pattern_remove_comments, "", stan_code)
        stan_code = stan_code.split("generated quantities")[-1]
        dtypes = re.findall(pattern_int, stan_code)
        dtypes = {item.strip() : 'int' for item in dtypes if item.strip() in self.varnames}
        return dtypes
