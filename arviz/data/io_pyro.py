"""Pyro-specific conversion code."""
import logging
import numpy as np

from .inference_data import InferenceData
from .base import requires, dict_to_dataset
from .. import utils

_log = logging.getLogger(__name__)


class PyroConverter:
    """Encapsulate Pyro specific logic."""

    nchains = None  # type: int
    ndraws = None  # type: int

    def __init__(
        self, *, posterior=None, prior=None, posterior_predictive=None, coords=None, dims=None
    ):
        """Convert Pyro data into an InferenceData object.

        Parameters
        ----------
        posterior : pyro.infer.MCMC
            Fitted MCMC object from Pyro
        prior: dict
            Prior samples from a Pyro model
        posterior_predictive : dict
            Posterior predictive samples for the posterior
        coords : dict[str] -> list[str]
            Map of dimensions to coordinates
        dims : dict[str] -> list[str]
            Map variable names to their coordinates
        """
        self.posterior = posterior
        if posterior is not None:
            self.nchains, self.ndraws = posterior.num_chains, posterior.num_samples
        else:
            self.nchains = self.ndraws = 0
        self.prior = prior
        self.posterior_predictive = posterior_predictive
        self.coords = coords
        self.dims = dims
        import pyro

        self.pyro = pyro

    @requires("posterior")
    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        data = self.posterior.get_samples(group_by_chain=True)
        data = {k: v.detach().cpu().numpy() for k, v in data.items()}
        return dict_to_dataset(data, library=self.pyro, coords=self.coords, dims=self.dims)

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from Pyro posterior."""
        divergences = self.posterior.diagnostics()["divergences"]
        diverging = np.zeros((self.nchains, self.ndraws), dtype=np.bool)
        for i, k in enumerate(sorted(divergences)):
            diverging[i, divergences[k]] = True
        data = {"diverging": diverging}
        return dict_to_dataset(data, library=self.pyro, coords=self.coords, dims=self.dims)

    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        data = {}
        for k, ary in self.posterior_predictive.items():
            ary = ary.detach().cpu().numpy()
            shape = ary.shape
            if shape[0] == self.nchains and shape[1] == self.ndraws:
                data[k] = ary
            elif shape[0] == self.nchains * self.ndraws:
                data[k] = ary.reshape((self.nchains, self.ndraws, *shape[1:]))
            else:
                data[k] = utils.expand_dims(ary)
                _log.warning(
                    "posterior predictive shape not compatible with number of chains and draws. "
                    "This can mean that some draws or even whole chains are not represented."
                )
        return dict_to_dataset(data, library=self.pyro, coords=self.coords, dims=self.dims)

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        return dict_to_dataset(
            {k: utils.expand_dims(v.detach().cpu().numpy()) for k, v in self.prior.items()},
            library=self.pyro,
            coords=self.coords,
            dims=self.dims,
        )

    def to_inference_data(self):
        """Convert all available data to an InferenceData object."""
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "prior": self.prior_to_xarray(),
            }
        )


def from_pyro(posterior=None, *, prior=None, posterior_predictive=None, coords=None, dims=None):
    """Convert Pyro data into an InferenceData object.

    Parameters
    ----------
    posterior : pyro.infer.MCMC
        Fitted MCMC object from Pyro
    prior: dict
        Prior samples from a Pyro model
    posterior_predictive : dict
        Posterior predictive samples for the posterior
    coords : dict[str] -> list[str]
        Map of dimensions to coordinates
    dims : dict[str] -> list[str]
        Map variable names to their coordinates
    """
    return PyroConverter(
        posterior=posterior,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords=coords,
        dims=dims,
    ).to_inference_data()
