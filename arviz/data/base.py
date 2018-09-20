"""Low level converters usually used by other functions."""
import warnings
from copy import deepcopy

import numpy as np
import xarray as xr


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

def generate_dims_coords(shape, var_name, dims=None, coords=None, default_dims=None):
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

    dims, coords = generate_dims_coords(shape, var_name,
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
