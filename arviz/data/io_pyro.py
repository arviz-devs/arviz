"""pyro-specific conversion code."""
from .inference_data import InferenceData
from .base import dict_to_dataset
from .. import utils


class PyroConverter:
    """Encapsulate Pyro specific logic."""

    def __init__(self, *, posterior, prior=None, posterior_predictive=None, observed_data=None, coords=None, dims=None):
        """Convert pyro data into an InferenceData object.

        Parameters
        ----------
        posterior : pyro.infer.MCMC
            Fitted MCMC object from Pyro
        prior: dict
            Prior samples from a Pyro model
        posterior_predictive : dict
            Posterior predictive samples for the posterior
        observed_data : dict
            Observed data used in the sampling.
        coords : dict[str] -> list[str]
            Map of dimensions to coordinates
        dims : dict[str] -> list[str]
            Map variable names to their coordinates
        """
        self.posterior = posterior
        self.prior = prior
        self.posterior_predictive = posterior_predictive
        self.observed_data = observed_data
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
                name: utils.expand_dims(samples.enumerate_support().squeeze())
                if self.posterior.num_chains == 1
                else samples.enumerate_support().squeeze()
                for name, samples in self.posterior.marginal(
                    sites=self.latent_vars
                ).empirical.items()
            }
        except AttributeError:  # Use pyro<0.3 release syntax
            data = {}
            for var_name in self.latent_vars:
                # pylint: disable=no-member
                samples = EmpiricalMarginal(
                    self.posterior, sites=var_name
                ).get_samples_and_weights()[0]
                samples = samples.numpy().squeeze()
                data[var_name] = utils.expand_dims(samples)
        return dict_to_dataset(data, library=self.pyro, coords=self.coords, dims=self.dims)

    def to_inference_data(self):
        """Convert all available data to an InferenceData object."""
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
            }
        )


def from_pyro(posterior=None, *, prior=None, posterior_predictive=None, observed_data=None, coords=None, dims=None):
    """Convert Pyro data into an InferenceData object.

    Parameters
    ----------
    posterior : pyro.infer.MCMC
        Fitted MCMC object from Pyro
    prior: dict
        Prior samples from a Pyro model
    posterior_predictive : dict
        Posterior predictive samples for the posterior
    observed_data : dict
        Observed data used in the sampling.
    coords : dict[str] -> list[str]
        Map of dimensions to coordinates
    dims : dict[str] -> list[str]
        Map variable names to their coordinates
    """
    return PyroConverter(
        posterior=posterior,
        prior=prior,
        posterior_predictive=posterior_predictive,
        observed_data=observed_data,
        coords=coords,
        dims=dims,
    ).to_inference_data()
