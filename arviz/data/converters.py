"""High level conversion functions."""
import numpy as np
import xarray as xr

from .inference_data import InferenceData
from .base import dict_to_dataset
from .io_pymc3 import from_pymc3
from .io_pystan import from_pystan


def convert_to_inference_data(obj, *, group='posterior', coords=None, dims=None, **kwargs):
    r"""Convert a supported object to an InferenceData object.

    This function sends `obj` to the right conversion function. It is idempotent,
    in that it will return arviz.InferenceData objects unchanged.

    Parameters
    ----------
    obj : dict, str, np.ndarray, xr.Dataset, pystan fit, pymc3 trace
        A supported object to convert to InferenceData:
            | InferenceData: returns unchanged
            | str: Attempts to load the netcdf dataset from disk
            | pystan fit: Automatically extracts data
            | pymc3 trace: Automatically extracts data
            | xarray.Dataset: adds to InferenceData as only group
            | dict: creates an xarray dataset as the only group
            | numpy array: creates an xarray dataset as the only group, gives the
                         array an arbitrary name
    group : str
        If `obj` is a dict or numpy array, assigns the resulting xarray
        dataset to this group. Default: "posterior".
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable
    kwargs
        Rest of the supported keyword arguments transferred to conversion function.

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
        return from_pystan(fit=obj, coords=coords, dims=dims, **kwargs)
    elif obj.__class__.__name__ == 'MultiTrace':  # ugly, but doesn't make PyMC3 a requirement
        return from_pymc3(trace=obj, coords=coords, dims=dims, **kwargs)

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
