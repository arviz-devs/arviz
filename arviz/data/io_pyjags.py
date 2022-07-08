"""Convert PyJAGS sample dictionaries to ArviZ inference data objects."""
import typing as tp
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import xarray

from .inference_data import InferenceData

from ..rcparams import rcParams
from .base import dict_to_dataset


class PyJAGSConverter:
    """Encapsulate PyJAGS specific logic."""

    def __init__(
        self,
        *,
        posterior: tp.Optional[tp.Mapping[str, np.ndarray]] = None,
        prior: tp.Optional[tp.Mapping[str, np.ndarray]] = None,
        log_likelihood: tp.Optional[
            tp.Union[str, tp.List[str], tp.Tuple[str, ...], tp.Mapping[str, str]]
        ] = None,
        coords=None,
        dims=None,
        save_warmup: tp.Optional[bool] = None,
        warmup_iterations: int = 0,
    ) -> None:
        self.posterior: tp.Optional[tp.Mapping[str, np.ndarray]]
        self.log_likelihood: tp.Optional[tp.Dict[str, np.ndarray]]
        if log_likelihood is not None and posterior is not None:
            posterior_copy = dict(posterior)  # create a shallow copy of the dictionary

            if isinstance(log_likelihood, str):
                log_likelihood = [log_likelihood]
            if isinstance(log_likelihood, (list, tuple)):
                log_likelihood = {name: name for name in log_likelihood}

            self.log_likelihood = {
                obs_var_name: posterior_copy.pop(log_like_name)
                for obs_var_name, log_like_name in log_likelihood.items()
            }
            self.posterior = posterior_copy
        else:
            self.posterior = posterior
            self.log_likelihood = None
        self.prior = prior
        self.coords = coords
        self.dims = dims
        self.save_warmup = rcParams["data.save_warmup"] if save_warmup is None else save_warmup
        self.warmup_iterations = warmup_iterations

        import pyjags

        self.pyjags = pyjags

    def _pyjags_samples_to_xarray(
        self, pyjags_samples: tp.Mapping[str, np.ndarray]
    ) -> tp.Tuple[xarray.Dataset, xarray.Dataset]:
        data, data_warmup = get_draws(
            pyjags_samples=pyjags_samples,
            warmup_iterations=self.warmup_iterations,
            warmup=self.save_warmup,
        )

        return (
            dict_to_dataset(data, library=self.pyjags, coords=self.coords, dims=self.dims),
            dict_to_dataset(
                data_warmup,
                library=self.pyjags,
                coords=self.coords,
                dims=self.dims,
            ),
        )

    def posterior_to_xarray(self) -> tp.Optional[tp.Tuple[xarray.Dataset, xarray.Dataset]]:
        """Extract posterior samples from fit."""
        if self.posterior is None:
            return None

        return self._pyjags_samples_to_xarray(self.posterior)

    def prior_to_xarray(self) -> tp.Optional[tp.Tuple[xarray.Dataset, xarray.Dataset]]:
        """Extract posterior samples from fit."""
        if self.prior is None:
            return None

        return self._pyjags_samples_to_xarray(self.prior)

    def log_likelihood_to_xarray(self) -> tp.Optional[tp.Tuple[xarray.Dataset, xarray.Dataset]]:
        """Extract log likelihood samples from fit."""
        if self.log_likelihood is None:
            return None

        return self._pyjags_samples_to_xarray(self.log_likelihood)

    def to_inference_data(self):
        """Convert all available data to an InferenceData object."""
        # obs_const_dict = self.observed_and_constant_data_to_xarray()
        # predictions_const_data = self.predictions_constant_data_to_xarray()
        save_warmup = self.save_warmup and self.warmup_iterations > 0
        # self.posterior is not None

        idata_dict = {
            "posterior": self.posterior_to_xarray(),
            "prior": self.prior_to_xarray(),
            "log_likelihood": self.log_likelihood_to_xarray(),
            "save_warmup": save_warmup,
        }

        return InferenceData(**idata_dict)


def get_draws(
    pyjags_samples: tp.Mapping[str, np.ndarray],
    variables: tp.Optional[tp.Union[str, tp.Iterable[str]]] = None,
    warmup: bool = False,
    warmup_iterations: int = 0,
) -> tp.Tuple[tp.Mapping[str, np.ndarray], tp.Mapping[str, np.ndarray]]:
    """
    Convert PyJAGS samples dictionary to ArviZ format and split warmup samples.

    Parameters
    ----------
    pyjags_samples: a dictionary mapping variable names to NumPy arrays of MCMC
                    chains of samples with shape
                    (parameter_dimension, chain_length, number_of_chains)

    variables: the variables to extract from the samples dictionary
    warmup: whether or not to return warmup draws in data_warmup
    warmup_iterations: the number of warmup iterations if any

    Returns
    -------
    A tuple of two samples dictionaries in ArviZ format
    """
    data_warmup: tp.Mapping[str, np.ndarray] = OrderedDict()

    if variables is None:
        variables = list(pyjags_samples.keys())
    elif isinstance(variables, str):
        variables = [variables]

    if not isinstance(variables, Iterable):
        raise TypeError("variables must be of type Sequence or str")

    variables = tuple(variables)

    if warmup_iterations > 0:
        (warmup_samples, actual_samples,) = _split_pyjags_dict_in_warmup_and_actual_samples(
            pyjags_samples=pyjags_samples,
            warmup_iterations=warmup_iterations,
            variable_names=variables,
        )

        data = _convert_pyjags_dict_to_arviz_dict(samples=actual_samples, variable_names=variables)

        if warmup:
            data_warmup = _convert_pyjags_dict_to_arviz_dict(
                samples=warmup_samples, variable_names=variables
            )
    else:
        data = _convert_pyjags_dict_to_arviz_dict(samples=pyjags_samples, variable_names=variables)

    return data, data_warmup


def _split_pyjags_dict_in_warmup_and_actual_samples(
    pyjags_samples: tp.Mapping[str, np.ndarray],
    warmup_iterations: int,
    variable_names: tp.Optional[tp.Tuple[str, ...]] = None,
) -> tp.Tuple[tp.Mapping[str, np.ndarray], tp.Mapping[str, np.ndarray]]:
    """
    Split a PyJAGS samples dictionary into actual samples and warmup samples.

    Parameters
    ----------
    pyjags_samples: a dictionary mapping variable names to NumPy arrays of MCMC
                    chains of samples with shape
                    (parameter_dimension, chain_length, number_of_chains)

    warmup_iterations: the number of draws to be split off for warmum
    variable_names: the variables in the dictionary to use; if None use all

    Returns
    -------
    A tuple of two pyjags samples dictionaries in PyJAGS format
    """
    if variable_names is None:
        variable_names = tuple(pyjags_samples.keys())

    warmup_samples: tp.Dict[str, np.ndarray] = {}
    actual_samples: tp.Dict[str, np.ndarray] = {}

    for variable_name, chains in pyjags_samples.items():
        if variable_name in variable_names:
            warmup_samples[variable_name] = chains[:, :warmup_iterations, :]
            actual_samples[variable_name] = chains[:, warmup_iterations:, :]

    return warmup_samples, actual_samples


def _convert_pyjags_dict_to_arviz_dict(
    samples: tp.Mapping[str, np.ndarray],
    variable_names: tp.Optional[tp.Tuple[str, ...]] = None,
) -> tp.Mapping[str, np.ndarray]:
    """
    Convert a PyJAGS dictionary to an ArviZ dictionary.

    Takes a python dictionary of samples that has been generated by the sample
    method of a model instance and returns a dictionary of samples in ArviZ
    format.

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

    variable_name_to_samples_map = {}

    if variable_names is None:
        variable_names = tuple(samples.keys())

    for variable_name, chains in samples.items():
        if variable_name in variable_names:
            parameter_dimension, _, _ = chains.shape
            if parameter_dimension == 1:
                variable_name_to_samples_map[variable_name] = chains[0, :, :].transpose()
            else:
                variable_name_to_samples_map[variable_name] = np.swapaxes(chains, 0, 2)

    return variable_name_to_samples_map


def _extract_arviz_dict_from_inference_data(
    idata,
) -> tp.Mapping[str, np.ndarray]:
    """
    Extract the samples dictionary from an ArviZ inference data object.

    Extracts a dictionary mapping parameter names to NumPy arrays of samples
    with shape (number_of_chains, chain_length, parameter_dimension) from an
    ArviZ inference data object.

    Parameters
    ----------
    idata: InferenceData

    Returns
    -------
    a dictionary mapping variable names to NumPy arrays with shape
             (number_of_chains, chain_length, parameter_dimension)

    """
    variable_name_to_samples_map = {
        key: np.array(value["data"])
        for key, value in idata.posterior.to_dict()["data_vars"].items()
    }

    return variable_name_to_samples_map


def _convert_arviz_dict_to_pyjags_dict(
    samples: tp.Mapping[str, np.ndarray]
) -> tp.Mapping[str, np.ndarray]:
    """
    Convert and ArviZ dictionary to a PyJAGS dictionary.

    Takes a python dictionary of samples in ArviZ format and returns the samples
    as a dictionary in PyJAGS format.

    Parameters
    ----------
    samples: dict of {str : array_like}
        a dictionary mapping variable names to NumPy arrays with shape
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

    variable_name_to_samples_map = {}

    for variable_name, chains in samples.items():
        if chains.ndim == 2:
            number_of_chains, chain_length = chains.shape
            chains = chains.reshape((number_of_chains, chain_length, 1))

        variable_name_to_samples_map[variable_name] = np.swapaxes(chains, 0, 2)

    return variable_name_to_samples_map


def from_pyjags(
    posterior: tp.Optional[tp.Mapping[str, np.ndarray]] = None,
    prior: tp.Optional[tp.Mapping[str, np.ndarray]] = None,
    log_likelihood: tp.Optional[tp.Mapping[str, str]] = None,
    coords=None,
    dims=None,
    save_warmup=None,
    warmup_iterations: int = 0,
) -> InferenceData:
    """
    Convert PyJAGS posterior samples to an ArviZ inference data object.

    Takes a python dictionary of samples that has been generated by the sample
    method of a model instance and returns an Arviz inference data object.
    For a usage example read the
    :ref:`Creating InferenceData section on from_pyjags <creating_InferenceData>`

    Parameters
    ----------
    posterior: dict of {str : array_like}, optional
        a dictionary mapping variable names to NumPy arrays containing
        posterior samples with shape
        (parameter_dimension, chain_length, number_of_chains)

    prior: dict of {str : array_like}, optional
        a dictionary mapping variable names to NumPy arrays containing
        prior samples with shape
        (parameter_dimension, chain_length, number_of_chains)

    log_likelihood: dict of {str: str}, list of str or str, optional
        Pointwise log_likelihood for the data. log_likelihood is extracted from the
        posterior. It is recommended to use this argument as a dictionary whose keys
        are observed variable names and its values are the variables storing log
        likelihood arrays in the JAGS code. In other cases, a dictionary with keys
        equal to its values is used.

    coords: dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.

    dims: dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable.

    save_warmup : bool, optional
       Save warmup iterations in InferenceData. If not defined, use default defined by the rcParams.

    warmup_iterations: int, optional
        Number of warmup iterations

    Returns
    -------
    InferenceData
    """
    return PyJAGSConverter(
        posterior=posterior,
        prior=prior,
        log_likelihood=log_likelihood,
        dims=dims,
        coords=coords,
        save_warmup=save_warmup,
        warmup_iterations=warmup_iterations,
    ).to_inference_data()
