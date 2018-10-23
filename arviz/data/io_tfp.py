"""Tfp-specific conversion code."""
import numpy as np

from .inference_data import InferenceData
from .base import dict_to_dataset


class TfpConverter:
    """Encapsulate tfp specific logic."""

    def __init__(self, posterior, *_, var_names=None, coords=None, dims=None):
        self.posterior = posterior

        if var_names is None:
            self.var_names = []
            for i in range(0, len(posterior)):
                self.var_names.append("var_{0}".format(i))
        else:
            self.var_names = var_names

        self.coords = coords
        self.dims = dims

        import tensorflow_probability as tfp

        self.tfp = tfp

    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        data = {}
        for i, var_name in enumerate(self.var_names):
            data[var_name] = np.expand_dims(self.posterior[i], axis=0)
        return dict_to_dataset(data, library=self.tfp, coords=self.coords, dims=self.dims)

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `trace`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        return InferenceData(**{"posterior": self.posterior_to_xarray()})


def from_tfp(posterior, var_names=None, *, coords=None, dims=None):
    """Convert tfp data into an InferenceData object."""
    return TfpConverter(
        posterior=posterior, var_names=var_names, coords=coords, dims=dims
    ).to_inference_data()
