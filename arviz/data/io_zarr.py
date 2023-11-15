"""Input and output support for zarr data."""

from .converters import convert_to_inference_data
from .inference_data import InferenceData


def from_zarr(store):
    return InferenceData.from_zarr(store)


from_zarr.__doc__ = InferenceData.from_zarr.__doc__


def to_zarr(data, store=None, **kwargs):
    """
    Convert data to zarr, optionally saving to disk if ``store`` is provided.

    The zarr storage is using the same group names as the InferenceData.

    Parameters
    ----------
    store : zarr.storage, MutableMapping or str, optional
        Zarr storage class or path to desired DirectoryStore.
        Default (None) a store is created in a temporary directory.
    **kwargs : dict, optional
        Passed to :py:func:`convert_to_inference_data`.

    Returns
    -------
    zarr.hierarchy.group
        A zarr hierarchy group containing the InferenceData.

    Raises
    ------
    TypeError
        If no valid store is found.


    References
    ----------
    https://zarr.readthedocs.io/

    """
    inference_data = convert_to_inference_data(data, **kwargs)
    zarr_group = inference_data.to_zarr(store=store)
    return zarr_group
