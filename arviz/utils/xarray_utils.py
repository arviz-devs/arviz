import re
import numpy as np
import xarray as xr

from arviz.compat import pymc3 as pm


def pymc3_to_xarray(trace, coords=None, dims=None):
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

    varnames, coords, dims = default_varnames_coords_dims(trace, coords, dims)

    verified, warning = verify_coords_dims(varnames, trace, coords, dims)

    data = xr.Dataset(coords=coords)
    base_dims = ['chain', 'draw']
    for key in varnames:
        vals = trace.get_values(key, combine=False, squeeze=False)
        vals = np.array(vals)
        dims_str = base_dims + dims[key]
        try:
            data[key] = xr.DataArray(vals, coords={v: coords[v] for v in dims_str}, dims=dims_str)
        except KeyError as exc:
            if not verified:
                raise TypeError(warning) from exc
            else:
                raise exc

    return data


def default_varnames_coords_dims(trace, coords, dims):
    """Set up varnames, coordinates, and dimensions for .to_xarray function

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
    iterable[str]
        The non-transformed variable names from the trace
    dict[str, iterable]
        Default coordinates for the trace
    dict[str, Tuple(str)]
        Default dimensions for the xarray
    """
    varnames = pm.utils.get_default_varnames(trace.varnames, include_transformed=False)
    if coords is None:
        coords = {}

    coords['draw'] = np.arange(len(trace))
    coords['chain'] = np.arange(trace.nchains)
    coords = {key: xr.IndexVariable((key,), data=vals) for key, vals in coords.items()}

    if dims is None:
        dims = {}

    for varname in varnames:
        dims.setdefault(varname, [])

    return varnames, coords, dims


def verify_coords_dims(varnames, trace, coords, dims):
    """Light checking and guessing on the structure of an xarray for a PyMC3 trace

    Parameters
    ----------
    varnames : iterable[string]
        list of dims for the xarray
    trace : pymc3.Multitrace
        trace from pymc3 run
    coords : dict
        output of `default_varnames_coords_dims`
    dims : dict
        output of `default_varnames_coords_dims`

    Returns
    -------
    bool
        Whether it passes the check
    str
        Warning string in case it does not pass
    """
    inferred_coords = coords.copy()
    inferred_dims = dims.copy()
    for key in ('draw', 'chain'):
        inferred_coords.pop(key)
    global_coords = {}
    throw = False

    for varname in varnames:
        vals = trace.get_values(varname, combine=False, squeeze=False)
        shapes = [d for shape in coords.values() for d in shape.shape]
        for idx, shape in enumerate(vals[0].shape[1:], 1):
            try:
                shapes.remove(shape)
            except ValueError:
                throw = True
                if shape not in global_coords:
                    global_coords[shape] = f'{varname}_dim_{idx}'
                key = global_coords[shape]
                inferred_dims[varname].append(key)
                if key not in inferred_coords:
                    inferred_coords[key] = f'np.arange({shape})'
    if throw:
        inferred_dims = {k: v for k, v in inferred_dims.items() if v}
        msg = 'Bad arguments! Try setting\ncoords={inferred_coords}\ndims={inferred_dims}'.format(
            inferred_coords=inferred_coords, inferred_dims=inferred_dims)
        return False, msg
    return True, ''


def pystan_to_xarray(fit, coords=None, dims=None):
    """Convert a PyStan StanFit4Model-object to an xarray dataset.

    Parameters
    ----------
    fit : StanFit4Model
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

    if fit.mode == 1:
        return "Stan model '{}' is of mode 'test_grad';\n"\
               "sampling is not conducted.".format(fit.model_name)
    elif fit.mode == 2:
        return "Stan model '{}' does not contain samples.".format(fit.model_name)

    varnames, coords, dims = pystan_varnames_coords_dims(fit, coords, dims)

    verified, warning = pystan_verify_coords_dims(varnames, fit, coords, dims)

    #infer dtypes
    pattern = r"int(?:\[.*\])*\s*(.)(?:\s*[=;]|(?:\s*<-))"
    generated_quantities = fit.get_stancode().split("generated quantities")[-1]
    dtypes = re.findall(pattern, generated_quantities)
    dtypes = {item : 'int' for item in dtypes if item in varnames}

    data = xr.Dataset(coords=coords)
    base_dims = ['chain', 'draw']
    for key in varnames:
        var_dtype = {key : 'int'} if key in dtypes else {}
        vals = fit.extract(key, dtypes=var_dtype, permuted=False)[key]
        if fit.sim['chains'] == 1:
            vals = np.expand_dims(vals, axis=1)
        dims_str = base_dims + dims[key]
        try:
            data[key] = xr.DataArray(vals, coords={v: coords[v] for v in dims_str}, dims=dims_str)
        except KeyError as exc:
            if not verified:
                raise TypeError(warning) from exc
            else:
                raise exc

    return data

def pystan_varnames_coords_dims(fit, coords, dims):
    """Set up varnames, coordinates, and dimensions for .to_xarray function

    fit : StanFit4Model
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
    varnames = fit.model_pars
    if coords is None:
        coords = {}

    coords['draw'] = np.arange(len(fit))
    coords['chain'] = np.arange(fit.sim['chains'])
    coords = {key: xr.IndexVariable((key,), data=vals) for key, vals in coords.items()}

    if dims is None:
        dims = {}

    for varname in varnames:
        dims.setdefault(varname, [])

    return varnames, coords, dims

def pystan_verify_coords_dims(varnames, fit, coords, dims):
    """Light checking and guessing on the structure of an xarray for a PyMC3 trace

    Parameters
    ----------
    varnames : iterable[string]
        list of dims for the xarray
    fit : StanFit4Model
        fit from PyStan sampling
    coords : dict
        output of `default_varnames_coords_dims`
    dims : dict
        output of `default_varnames_coords_dims`

    Returns
    -------
    bool
        Whether it passes the check
    str
        Warning string in case it does not pass
    """
    inferred_coords = coords.copy()
    inferred_dims = dims.copy()
    for key in ('draw', 'chain'):
        inferred_coords.pop(key)
    global_coords = {}
    throw = False

    for varname in varnames:
        vals = fit.extract(key, dtypes=var_dtype, permuted=False)[key]
        if fit.sim['chains'] == 1:
            vals = np.expand_dims(vals, axis=1)
        shapes = [d for shape in coords.values() for d in shape.shape]
        for idx, shape in enumerate(vals[0].shape[1:], 1):
            try:
                shapes.remove(shape)
            except ValueError:
                throw = True
                if shape not in global_coords:
                    global_coords[shape] = f'{varname}_dim_{idx}'
                key = global_coords[shape]
                inferred_dims[varname].append(key)
                if key not in inferred_coords:
                    inferred_coords[key] = f'np.arange({shape})'
    if throw:
        inferred_dims = {k: v for k, v in inferred_dims.items() if v}
        msg = 'Bad arguments! Try setting\ncoords={inferred_coords}\ndims={inferred_dims}'.format(
            inferred_coords=inferred_coords, inferred_dims=inferred_dims)
        return False, msg
    return True, ''
