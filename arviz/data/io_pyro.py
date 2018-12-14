"""pyro-specific conversion code."""
import numpy as np

from .inference_data import InferenceData
from .base import dict_to_dataset


def _get_var_names(posterior):
    """Extract latent and observed variable names from pyro.MCMC.

    Parameters
    ----------
    posterior : pyro.MCMC
        Fitted MCMC object from Pyro

    Returns
    -------
    list[str], list[str]
    observed and latent variable names from the MCMC trace.
    """
    sample_point = posterior.exec_traces[0]
    nodes = [node for node in sample_point.nodes.values() if node["type"] == "sample"]
    observed = [node["name"] for node in nodes if node["is_observed"]]
    latent = [node["name"] for node in nodes if not node["is_observed"]]
    return observed, latent


class PyroConverter:
    """Encapsulate Pyro specific logic."""

    def __init__(self, posterior, *_, coords=None, dims=None):
        """Convert pyro data into an InferenceData object.

        Parameters
        ----------
        posterior : pyro.MCMC
            Fitted MCMC object from Pyro
        coords : dict[str] -> list[str]
            Map of dimensions to coordinates
        dims : dict[str] -> list[str]
            Map variable names to their coordinates
        """
        self.posterior = posterior
        self.observed_vars, self.latent_vars = _get_var_names(posterior)
        self.coords = coords
        self.dims = dims
        import pyro

        self.pyro = pyro

    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        # Do not make pyro a requirement
        from pyro.infer import EmpiricalMarginal

        try:  # Try pyro>=0.3 release syntax
            data = {
                name: np.expand_dims(samples.enumerate_support(), 0)
                for name, samples in self.posterior.marginal(
                    sites=self.latent_vars
                ).empirical.items()
            }
        except AttributeError:  # Use pyro<0.3 release syntax
            data = {}
            for var_name in self.latent_vars:
                samples = EmpiricalMarginal(
                    self.posterior, sites=var_name
                ).get_samples_and_weights()[0]
                data[var_name] = np.expand_dims(samples.numpy().squeeze(), 0)
        return dict_to_dataset(data, library=self.pyro, coords=self.coords, dims=self.dims)

    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        from pyro.infer import EmpiricalMarginal

        try:  # Try pyro>=0.3 release syntax
            data = {
                name: np.expand_dims(samples.enumerate_support(), 0)
                for name, samples in self.posterior.marginal(
                    sites=self.observed_vars
                ).empirical.items()
            }
        except AttributeError:  # Use pyro<0.3 release syntax
            data = {}
            for var_name in self.observed_vars:
                samples = EmpiricalMarginal(
                    self.posterior, sites=var_name
                ).get_samples_and_weights()[0]
                data[var_name] = np.expand_dims(samples.numpy().squeeze(), 0)
        return dict_to_dataset(data, library=self.pyro, coords=self.coords, dims=self.dims)

    def to_inference_data(self):
        """Convert all available data to an InferenceData object."""
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
            }
        )


def from_pyro(posterior, *, coords=None, dims=None):
    """Convert pyro data into an InferenceData object.

    Parameters
    ----------
    posterior : pyro.MCMC
        Fitted MCMC object from Pyro
    coords : dict[str] -> list[str]
        Map of dimensions to coordinates
    dims : dict[str] -> list[str]
        Map variable names to their coordinates
    """
    return PyroConverter(posterior=posterior, coords=coords, dims=dims).to_inference_data()
