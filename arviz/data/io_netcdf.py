"""Input and output support for data."""

from .converters import convert_to_inference_data
from .inference_data import InferenceData


def from_netcdf(filename):
    """Load netcdf file back into an arviz.InferenceData.

    Parameters
    ----------
    filename : str
        name or path of the file to load trace

    Returns
    -------
        InferenceData object

    Notes
    -----
    By default, the datasets of the InferenceData object will be lazily loaded instead
    of loaded into memory. This behaviour is regulated by the value of
    ``az.rcParams["data.load"]``.
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
    file_name = inference_data.to_netcdf(filename)
    return file_name
