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
        return False, f'Bad arguments! Try setting\ncoords={inferred_coords}\ndims={inferred_dims}'
    return True, ''
