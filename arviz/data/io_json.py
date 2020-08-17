"""Input and output support for data."""

from .converters import convert_to_inference_data
from .io_dict import from_dict

try:
    import ujson as json
except ImportError:
    # Can't find ujson using json
    import json


def from_json(filename):
    """Initialize object from a json file.

    Parameters
    ----------
    filename : str
        location of json file

    Returns
    -------
    InferenceData object
    """
    with open(filename, "rb") as file:
        idata_dict = json.load(file)

    return from_dict(**idata_dict, save_warmup=True)


def to_json(data, filename, *, group="posterior", coords=None, dims=None):
    """Save dataset as a json file.

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
    file_name = inference_data.to_json(filename)
    return file_name
