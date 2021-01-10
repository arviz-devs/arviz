"""Input and output support for data."""

from .io_dict import from_dict

try:
    import ujson as json
except ImportError:
    # Can't find ujson using json
    # mypy struggles with conditional imports expressed as catching ImportError:
    # https://github.com/python/mypy/issues/1153
    import json  # type: ignore


def from_json(filename):
    """Initialize object from a json file.

    Will use the faster `ujson` (https://github.com/ultrajson/ultrajson) if it is available.

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


def to_json(idata, filename):
    """Save dataset as a json file.

    Will use the faster `ujson` (https://github.com/ultrajson/ultrajson) if it is available.

    WARNING: Only idempotent in case `idata` is InferenceData.

    Parameters
    ----------
    idata : InferenceData
        Object to be saved
    filename : str
        name or path of the file to load trace

    Returns
    -------
    str
        filename saved to
    """
    file_name = idata.to_json(filename)
    return file_name
