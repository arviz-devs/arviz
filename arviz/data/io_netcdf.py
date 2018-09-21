"""Input and output support for data."""
import os

from .inference_data import InferenceData
from .converters import convert_to_inference_data


def load_data(filename):
    """Load netcdf file back into an arviz.InferenceData.

    Parameters
    ----------
    filename : str
        name or path of the file to load trace
    """
    return InferenceData.from_netcdf(filename)


def save_data(data, filename, *, group='posterior', coords=None, dims=None):
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


def load_arviz_data(dataset):
    """Load built-in arviz dataset into memory.

    Available datasets are `centered_eight` and `non_centered_eight`. Will print out available
    datasets in case of error.

    Parameters
    ----------
    dataset : str
        Name of dataset to load

    Returns
    -------
    InferenceData
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(here, 'data', '_datasets')
    datasets_available = {
        'centered_eight': {
            'description': '''
                Centered eight schools model.  Four chains, 500 draws each, fit with
                NUTS in PyMC3.  Features named coordinates for each of the eight schools.
            ''',
            'path': os.path.join(data_path, 'centered_eight.nc')
        },
        'non_centered_eight': {
            'description': '''
                Non-centered eight schools model.  Four chains, 500 draws each, fit with
                NUTS in PyMC3.  Features named coordinates for each of the eight schools.
            ''',
            'path': os.path.join(data_path, 'non_centered_eight.nc')
        },
    }
    if dataset in datasets_available:
        return InferenceData.from_netcdf(datasets_available[dataset]['path'])
    else:
        msg = ['\'dataset\' must be one of the following options:']
        for key, value in sorted(datasets_available.items()):
            msg.append('{key}: {description}'.format(key=key, description=value['description']))

        raise ValueError('\n'.join(msg))
