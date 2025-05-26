"""Conversion between InferenceData and DataTree."""

from .inference_data import InferenceData


def to_datatree(data):
    """Convert InferenceData object to a :class:`~xarray.DataTree`.

    Parameters
    ----------
    data : InferenceData
    """
    return data.to_datatree()


def from_datatree(datatree):
    """Create an InferenceData object from a :class:`~xarray.DataTree`.

    Parameters
    ----------
    datatree : DataTree
    """
    return InferenceData.from_datatree(datatree)
