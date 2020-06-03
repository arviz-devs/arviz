import typing as tp
import numpy as np

import arviz as az


def _convert_pyjags_samples_dictionary_to_arviz_samples_dictionary(
        samples: tp.Dict[str, np.ndarray]) \
        -> tp.Dict[str, np.ndarray]:
    """
    This function takes a python dictionary of samples that has been generated
    by the sample method of a model instance and returns a dictionary of samples
    in ArviZ format.

    Parameters
    ----------
    samples: a dictionary mapping variable names to P arrays with shape
             (parameter_dimension, chain_length, number_of_chains)

    Returns
    -------
    a dictionary mapping variable names to NumPy arrays with shape
             (number_of_chains, chain_length, parameter_dimension)
    """

    # pyjags returns a dictionary of NumPy arrays with shape
    #         (parameter_dimension, chain_length, number_of_chains)
    # but arviz expects samples with shape
    #         (number_of_chains, chain_length, parameter_dimension)

    parameter_name_to_samples_map = {}

    for parameter_name, chains in samples.items():
        parameter_dimension, _, _ = chains.shape
        if parameter_dimension == 1:
            parameter_name_to_samples_map[parameter_name] = \
                chains[0, :, :].transpose()
        else:
            parameter_name_to_samples_map[parameter_name] = \
                np.swapaxes(chains, 0, 2)

    return parameter_name_to_samples_map


def _convert_pyjags_samples_dict_to_arviz_inference_data(
        samples: tp.Dict[str, np.ndarray]) -> az.InferenceData:
    """
    This function takes a python dictionary of samples that has been generated
    by the sample method of a model instance and returns an Arviz inference data
    object.
    Parameters
    ----------
    samples: a dictionary mapping variable names to NumPy arrays with shape
             (parameter_dimension, chain_length, number_of_chains)

    Returns
    -------
    An Arviz inference data object
    """

    # pyjags returns a dictionary of NumPy arrays with shape
    #         (parameter_dimension, chain_length, number_of_chains)
    # but arviz expects samples with shape
    #         (number_of_chains, chain_length, parameter_dimension)

    return az.convert_to_inference_data(
                _convert_pyjags_samples_dictionary_to_arviz_samples_dictionary(
                    samples))


def _extract_samples_dictionary_from_arviz_inference_data(idata) \
        -> tp.Dict[str, np.ndarray]:
    """
    This function extracts a dictionary mapping parameter names to NumPy arrays
    of samples with shape (number_of_chains, chain_length, parameter_dimension)
    from an ArviZ inference data object.

    Parameters
    ----------
    idata: An Arviz inference data object

    Returns
    -------
    a dictionary mapping variable names to NumPy arrays with shape
             (number_of_chains, chain_length, parameter_dimension)

    """
    parameter_name_to_samples_map = {}

    for key, value in idata.posterior.to_dict()['data_vars'].items():
        parameter_name_to_samples_map[key] = np.array(value['data'])

    return parameter_name_to_samples_map


def _convert_arviz_samples_dictionary_to_pyjags_samples_dictionary(
        samples: tp.Dict[str, np.ndarray]) \
        -> tp.Dict[str, np.ndarray]:
    """
    This function takes a python dictionary of samples in ArviZ format
    and returns the samples as a dictionary in PyJAGS format.

    Parameters
    ----------
    samples: a dictionary mapping variable names to NumPy arrays with shape
             (number_of_chains, chain_length, parameter_dimension)

    Returns
    -------
    a dictionary mapping variable names to NumPy arrays with shape
             (parameter_dimension, chain_length, number_of_chains)

    """

    # pyjags returns a dictionary of NumPy arrays with shape
    #         (parameter_dimension, chain_length, number_of_chains)
    # but arviz expects samples with shape
    #         (number_of_chains, chain_length, parameter_dimension)

    parameter_name_to_samples_map = {}

    for parameter_name, chains in samples.items():
        if chains.ndim == 2:
            number_of_chains, chain_length = chains.shape
            chains = chains.reshape((number_of_chains, chain_length, 1))

        parameter_name_to_samples_map[parameter_name] = \
            np.swapaxes(chains, 0, 2)

    return parameter_name_to_samples_map


def from_pyjags(posterior: tp.Dict[str, np.ndarray]) -> az.InferenceData:
    return _convert_pyjags_samples_dict_to_arviz_inference_data(posterior)
