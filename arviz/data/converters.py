"""High level conversion functions."""
import numpy as np
import xarray as xr

from .base import dict_to_dataset
from .inference_data import InferenceData
from .io_beanmachine import from_beanmachine
from .io_cmdstan import from_cmdstan
from .io_cmdstanpy import from_cmdstanpy
from .io_emcee import from_emcee
from .io_numpyro import from_numpyro
from .io_pyro import from_pyro
from .io_pystan import from_pystan


# pylint: disable=too-many-return-statements
def convert_to_inference_data(obj, *, group="posterior", coords=None, dims=None, **kwargs):
    r"""Convert a supported object to an InferenceData object.

    This function sends `obj` to the right conversion function. It is idempotent,
    in that it will return arviz.InferenceData objects unchanged.

    Parameters
    ----------
    obj : dict, str, np.ndarray, xr.Dataset, pystan fit
        A supported object to convert to InferenceData:
            | InferenceData: returns unchanged
            | str: Attempts to load the cmdstan csv or netcdf dataset from disk
            | pystan fit: Automatically extracts data
            | cmdstanpy fit: Automatically extracts data
            | cmdstan csv-list: Automatically extracts data
            | emcee sampler: Automatically extracts data
            | pyro MCMC: Automatically extracts data
            | beanmachine MonteCarloSamples: Automatically extracts data
            | xarray.Dataset: adds to InferenceData as only group
            | xarray.DataArray: creates an xarray dataset as the only group, gives the
                         array an arbitrary name, if name not set
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
    kwargs[group] = obj
    kwargs["coords"] = coords
    kwargs["dims"] = dims

    # Cases that convert to InferenceData
    if isinstance(obj, InferenceData):
        if coords is not None or dims is not None:
            raise TypeError("Cannot use coords or dims arguments with InferenceData value.")
        return obj
    elif isinstance(obj, str):
        if obj.endswith(".csv"):
            if group == "sample_stats":
                kwargs["posterior"] = kwargs.pop(group)
            elif group == "sample_stats_prior":
                kwargs["prior"] = kwargs.pop(group)
            return from_cmdstan(**kwargs)
        else:
            if coords is not None or dims is not None:
                raise TypeError(
                    "Cannot use coords or dims arguments reading InferenceData from netcdf."
                )
            return InferenceData.from_netcdf(obj)
    elif (
        obj.__class__.__name__ in {"StanFit4Model", "CmdStanMCMC"}
        or obj.__class__.__module__ == "stan.fit"
    ):
        if group == "sample_stats":
            kwargs["posterior"] = kwargs.pop(group)
        elif group == "sample_stats_prior":
            kwargs["prior"] = kwargs.pop(group)
        if obj.__class__.__name__ == "CmdStanMCMC":
            return from_cmdstanpy(**kwargs)
        else:  # pystan or pystan3
            return from_pystan(**kwargs)
    elif obj.__class__.__name__ == "EnsembleSampler":  # ugly, but doesn't make emcee a requirement
        return from_emcee(sampler=kwargs.pop(group), **kwargs)
    elif obj.__class__.__name__ == "MonteCarloSamples":
        return from_beanmachine(sampler=kwargs.pop(group), **kwargs)
    elif obj.__class__.__name__ == "MCMC" and obj.__class__.__module__.startswith("pyro"):
        return from_pyro(posterior=kwargs.pop(group), **kwargs)
    elif obj.__class__.__name__ == "MCMC" and obj.__class__.__module__.startswith("numpyro"):
        return from_numpyro(posterior=kwargs.pop(group), **kwargs)

    # Cases that convert to xarray
    if isinstance(obj, xr.Dataset):
        dataset = obj
    elif isinstance(obj, xr.DataArray):
        if obj.name is None:
            obj.name = "x"
        dataset = obj.to_dataset()
    elif isinstance(obj, dict):
        dataset = dict_to_dataset(obj, coords=coords, dims=dims)
    elif isinstance(obj, np.ndarray):
        dataset = dict_to_dataset({"x": obj}, coords=coords, dims=dims)
    elif isinstance(obj, (list, tuple)) and isinstance(obj[0], str) and obj[0].endswith(".csv"):
        if group == "sample_stats":
            kwargs["posterior"] = kwargs.pop(group)
        elif group == "sample_stats_prior":
            kwargs["prior"] = kwargs.pop(group)
        return from_cmdstan(**kwargs)
    else:
        allowable_types = (
            "xarray dataarray",
            "xarray dataset",
            "dict",
            "netcdf filename",
            "numpy array",
            "pystan fit",
            "emcee fit",
            "pyro mcmc fit",
            "numpyro mcmc fit",
            "cmdstan fit csv filename",
            "cmdstanpy fit",
        )
        raise ValueError(
            f'Can only convert {", ".join(allowable_types)} to InferenceData, '
            f"not {obj.__class__.__name__}"
        )

    return InferenceData(**{group: dataset})


def convert_to_dataset(obj, *, group="posterior", coords=None, dims=None):
    """Convert a supported object to an xarray dataset.

    This function is idempotent, in that it will return xarray.Dataset functions
    unchanged. Raises `ValueError` if the desired group can not be extracted.

    Note this goes through a DataInference object. See `convert_to_inference_data`
    for more details. Raises ValueError if it can not work out the desired
    conversion.

    Parameters
    ----------
    obj : dict, str, np.ndarray, xr.Dataset, pystan fit
        A supported object to convert to InferenceData:

        - InferenceData: returns unchanged
        - str: Attempts to load the netcdf dataset from disk
        - pystan fit: Automatically extracts data
        - xarray.Dataset: adds to InferenceData as only group
        - xarray.DataArray: creates an xarray dataset as the only group, gives the
          array an arbitrary name, if name not set
        - dict: creates an xarray dataset as the only group
        - numpy array: creates an xarray dataset as the only group, gives the
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
        raise ValueError(
            "Can not extract {group} from {obj}! See {filename} for other "
            "conversion utilities.".format(group=group, obj=obj, filename=__file__)
        )
    return dataset
