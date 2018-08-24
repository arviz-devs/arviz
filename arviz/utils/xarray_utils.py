from abc import ABC, abstractmethod, abstractstaticmethod
import re
from copy import deepcopy as copy
import itertools

import numpy as np
import xarray as xr

from arviz.compat import pymc3 as pm


def get_converter(obj, coords=None, dims=None, chains=None):
    """Get the converter to transform a supported object to an xarray dataset.

    This function sends `obj` to the right conversion function. It is idempotent,
    in that it will return xarray.Datasets unchanged.

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
    xarray.Dataset
        The coordinates are those passed in and ('chain', 'draw')
    """
    if isinstance(obj, dict):
        return DictToXarray(obj, coords, dims, chains=chains)
    elif obj.__class__.__name__ == 'StanFit4Model':  # ugly, but doesn't make PyStan a requirement
        return PyStanToXarray(obj, coords, dims)
    elif obj.__class__.__name__ == 'MultiTrace':  # ugly, but doesn't make PyMC3 a requirement
        return PyMC3ToXarray(obj, coords, dims)
    else:
        raise TypeError('Can only convert PyStan or PyMC3 object to xarray, not {}'.format(
            obj.__class__.__name__))


def convert_to_xarray(obj, coords=None, dims=None, chains=None):
    """Convert a supported object to an xarray dataset.

    This function sends `obj` to the right conversion function. It is idempotent,
    in that it will return xarray.Datasets unchanged.

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
    xarray.Dataset
        The coordinates are those passed in and ('chain', 'draw')
    """
    if isinstance(obj, xr.Dataset):
        return obj
    else:
        return get_converter(obj, coords=coords, dims=dims, chains=chains).to_xarray()


class Converter(ABC):
    def __init__(self, obj, coords=None, dims=None):
        self.obj = obj
        # pylint: disable=assignment-from-none
        self.varnames, self.coords, self.dims = self.default_varnames_coords_dims(obj, coords, dims)
        self.verified, self.warning = self.verify_coords_dims()

    @abstractmethod
    def to_xarray(self):
        pass

    @abstractstaticmethod
    def default_varnames_coords_dims(obj, coords, dims):  # pylint: disable=unused-argument
        return

    @abstractmethod
    def verify_coords_dims(self):
        return


class DictToXarray(Converter):
    def __init__(self, obj, coords=None, dims=None, chains=None):
        """Convert a dict containing posterior samples to an xarray dataset.

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

        Returns
        -------
        xarray.Dataset
            The coordinates are those passed in and ('chain', 'draw')
        """
        if chains is None:
            raise ValueError('Number of chains required when converting a dict')

        if coords is None:
            coords = {}

        coords['chain'] = np.arange(chains)
        super().__init__(obj, coords=coords, dims=dims)

    def to_xarray(self):
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
        """Set up varnames, coordinates, and dimensions for .to_xarray function

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


class PyMC3ToXarray(Converter):
    def __init__(self, trace, coords=None, dims=None):
        """Convert a pymc3 trace to an xarray dataset.

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
        xarray.Dataset
            The coordinates are those passed in and ('chain', 'draw')
        """
        super().__init__(trace, coords=coords, dims=dims)

    def to_xarray(self):
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
        """Set up varnames, coordinates, and dimensions for .to_xarray function

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


class PyStanToXarray(Converter):
    def __init__(self, fit, coords=None, dims=None):
        """Convert a PyStan StanFit4Model-object to an xarray dataset.

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
        xarray.Dataset
            The coordinates are those passed in and ('chain', 'draw')
        """
        super().__init__(fit, coords=coords, dims=dims)

    def to_xarray(self):
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
        """Set up varnames, coordinates, and dimensions for .to_xarray function

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
        extract = obj.extract(varnames, permuted=False)
        for varname, vals in extract.items():
            if varname not in dims:
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
            return "Stan model '{}' is of mode 'test_grad';\n"\
                "sampling is not conducted.".format(fit.model_name)
        elif fit.mode == 2:
            return "Stan model '{}' does not contain samples.".format(fit.model_name)

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


def xarray_var_iter(data, var_names=None, combined=False, skip_dims=None, reverse_selections=False):
    """Converts xarray data to an iterator over vectors
    Iterates over each var_name and all of its coordinates, returning the 1d
    data
    Parameters
    ----------
    data : xarray.Dataset
        Posterior data in an xarray
    var_names : iterator of strings (optional)
        Should be a subset of data.data_vars. Defaults to all of them.
    combined : bool
        Whether to combine chains or leave them separate
    skip_dims : set
        dimensions to not iterate over
    reverse_selections : bool
        Whether to reverse selections before iterating.
    Returns
    -------
    Iterator of (str, dict(str, any), np.array)
        The string is the variable name, the dictionary are coordinate names to values,
        and the array are the values of the variable at those coordinates.
    """
    if skip_dims is None:
        skip_dims = set()

    if combined:
        skip_dims = skip_dims.union({'chain', 'draw'})
    else:
        skip_dims.add('draw')

    if var_names is None:
        if isinstance(data, xr.Dataset):
            var_names = list(data.data_vars)
        elif isinstance(data, xr.DataArray):
            var_names = [data.name]
            data = {data.name: data}

    for var_name in var_names:
        if var_name in data:
            new_dims = set(data[var_name].dims) - skip_dims
            vals = [data[var_name][dim].values for dim in new_dims]
            dims = [{k: v for k, v in zip(new_dims, prod)} for prod in itertools.product(*vals)]
            if reverse_selections:
                dims = reversed(dims)

            for selection in dims:
                yield var_name, selection, data[var_name].sel(**selection).values

def selection_to_string(selection):
    """Convert dictionary of coordinates to a string for labels.

    Parameters
    ----------
    selection : dict[Any] -> Any

    Returns
    -------
    str
        key1: value1, key2: value2, ...
    """
    return ', '.join(['{}: {}'.format(k, v) for k, v in selection.items()])
