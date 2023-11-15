# pylint: disable=arguments-differ
"""Base class for PyMC interface wrappers."""
from .base import SamplingWrapper


# pylint: disable=abstract-method
class PyMCSamplingWrapper(SamplingWrapper):
    """PyMC (4.0+) sampling wrapper base class.

    See the documentation on  :class:`~arviz.SamplingWrapper` for a more detailed
    description. An example of ``PyMCSamplingWrapper`` usage can be found
    in the :ref:`pymc_refitting` notebook.

    Warnings
    --------
    Sampling wrappers are an experimental feature in a very early stage. Please use them
    with caution.
    """

    def sample(self, modified_observed_data):
        """Update data and sample model on modified_observed_data."""
        import pymc  # pylint: disable=import-error

        with self.model:
            pymc.set_data(modified_observed_data)
            idata = pymc.sample(
                **self.sample_kwargs,
            )
        return idata

    def get_inference_data(self, fitted_model):
        """Return sampling result without modifying.

        PyMC sampling already returns and InferenceData object.
        """
        return fitted_model
