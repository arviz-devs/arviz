# pylint: disable=arguments-differ
"""Base class for PyStan wrappers."""
from typing import Union

from ..data import from_cmdstanpy, from_pystan
from .base import SamplingWrapper


# pylint: disable=abstract-method
class StanSamplingWrapper(SamplingWrapper):
    """Stan sampling wrapper base class.

    See the documentation on  :class:`~arviz.SamplingWrapper` for a more detailed
    description. An example of ``PyStanSamplingWrapper`` usage can be found
    in the :ref:`pystan_refitting` notebook. For usage examples of other wrappers
    see the user guide pages on :ref:`wrapper_guide`.

    Warnings
    --------
    Sampling wrappers are an experimental feature in a very early stage. Please use them
    with caution.

    See Also
    --------
    SamplingWrapper
    """

    def sel_observations(self, idx):
        """Select a subset of the observations in idata_orig.

        **Not implemented**: This method must be implemented on a model basis.
        It is documented here to show its format and call signature.

        Parameters
        ----------
        idx
            Indexes to separate from the rest of the observed data.

        Returns
        -------
        modified_observed_data : dict
            Dictionary containing both excluded and included data but properly divided
            in the different keys. Passed to ``data`` argument of ``model.sampling``.
        excluded_observed_data : str
            Variable name containing the pointwise log likelihood data of the excluded
            data. As PyStan cannot call C++ functions and log_likelihood__i is already
            calculated *during* the simultion, instead of the value on which to evaluate
            the likelihood, ``log_likelihood__i`` expects a string so it can extract the
            corresponding data from the InferenceData object.
        """
        raise NotImplementedError("sel_observations must be implemented on a model basis")

    def get_inference_data(self, fitted_model):  # pylint: disable=arguments-renamed
        """Convert the fit object returned by ``self.sample`` to InferenceData."""
        if fitted_model.__class__.__name__ == "CmdStanMCMC":
            idata = from_cmdstanpy(posterior=fitted_model, **self.idata_kwargs)
        else:
            idata = from_pystan(posterior=fitted_model, **self.idata_kwargs)
        return idata

    def log_likelihood__i(self, excluded_obs, idata__i):
        """Retrieve the log likelihood of the excluded observations from ``idata__i``."""
        return idata__i.log_likelihood[excluded_obs]


class PyStan2SamplingWrapper(StanSamplingWrapper):
    """PyStan (2.x) sampling wrapper base class.

    See the documentation on  :class:`~arviz.SamplingWrapper` for a more detailed
    description. An example of ``PyStanSamplingWrapper`` usage can be found
    in the :ref:`pystan_refitting` notebook. For usage examples of other wrappers
    see the user guide pages on :ref:`wrapper_guide`.

    Warnings
    --------
    Sampling wrappers are an experimental feature in a very early stage. Please use them
    with caution.

    See Also
    --------
    SamplingWrapper
    """

    def sample(self, modified_observed_data):
        """Resample the PyStan model stored in self.model on modified_observed_data."""
        fit = self.model.sampling(data=modified_observed_data, **self.sample_kwargs)
        return fit


class PyStanSamplingWrapper(StanSamplingWrapper):
    """PyStan (3.0+) sampling wrapper base class.

    See the documentation on  :class:`~arviz.SamplingWrapper` for a more detailed
    description. An example of ``PyStan3SamplingWrapper`` usage can be found
    in the :ref:`pystan3_refitting` notebook.

    Warnings
    --------
    Sampling wrappers are an experimental feature in a very early stage. Please use them
    with caution.
    """

    def sample(self, modified_observed_data):
        """Rebuild and resample the PyStan model on modified_observed_data."""
        import stan  # pylint: disable=import-error,import-outside-toplevel

        self.model: Union[str, stan.Model]
        if isinstance(self.model, str):
            program_code = self.model
        else:
            program_code = self.model.program_code
        self.model = stan.build(program_code, data=modified_observed_data)
        fit = self.model.sample(**self.sample_kwargs)
        return fit
