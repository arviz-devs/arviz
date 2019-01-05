"""Input and output support for data."""
import warnings
from .inference_data import InferenceData
from .converters import convert_to_inference_data


def from_netcdf(filename):
    """Load netcdf file back into an arviz.InferenceData.

    Parameters
    ----------
    filename : str
        name or path of the file to load trace
    """
    return InferenceData.from_netcdf(filename)


def to_netcdf(data, filename, *, group="posterior", coords=None, dims=None):
    """Save dataset as a netcdf file.

    WARNING: Only idempotent in case `data` is InferenceData

    Parameters
    ----------
    data : InferenceData, or any object accepted by `convert_to_inference_data`
        Object to be saved
    filename : str
        name or path of the file to load trace
    group : str (optional)
        In case `data` is not InferenceData, this is the group it will be saved to
    coords : dict (optional)
        See `convert_to_inference_data`
    dims : dict (optional)
        See `convert_to_inference_data`

    Returns
    -------
    str
        filename saved to
    """
    inference_data = convert_to_inference_data(data, group=group, coords=coords, dims=dims)
    return inference_data.to_netcdf(filename)


def load_data(filename):
    """Load netcdf file back into an arviz.InferenceData.

    Parameters
    ----------
    filename : str
        name or path of the file to load trace

    Note
    ----
    This function is deprecated and will be removed in 0.4.
    Use `from_netcdf` instead.
    """
    warnings.warn(
        "The 'load_data' function is deprecated as of 0.3.2, use 'from_netcdf' instead",
        DeprecationWarning,
    )
    return from_netcdf(filename=filename)


def save_data(data, filename, *, group="posterior", coords=None, dims=None):
    """Save dataset as a netcdf file.

    WARNING: Only idempotent in case `data` is InferenceData

    Parameters
    ----------
    data : InferenceData, or any object accepted by `convert_to_inference_data`
        Object to be saved
    filename : str
        name or path of the file to load trace
    group : str (optional)
        In case `data` is not InferenceData, this is the group it will be saved to
    coords : dict (optional)
        See `convert_to_inference_data`
    dims : dict (optional)
        See `convert_to_inference_data`

    Returns
    -------
    str
        filename saved to

    Note
    ----
    This function is deprecated and will be removed in 0.4.
    Use `to_netcdf` instead.
    """
    warnings.warn(
        "The 'save_data' function is deprecated as of 0.3.2, use 'to_netcdf' instead",
        DeprecationWarning,
    )
    return to_netcdf(data=data, filename=filename, group=group, coords=coords, dims=dims)
