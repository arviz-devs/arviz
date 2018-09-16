"""Miscellaneous utilities for supporting ArviZ."""
import os

from ..inference_data import InferenceData


def load_data(filename):
    """Load netcdf file back into an arviz.InferenceData.

    Parameters
    ----------
    filename : str
        name or path of the file to load trace
    """
    return InferenceData(filename)


def load_arviz_data(dataset):
    """Load built-in arviz dataset into memory.

    Will print out available datasets in case of error.

    Parameters
    ----------
    dataset : str
        Name of dataset to load

    Returns
    -------
    InferenceData
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(here, 'data')
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
