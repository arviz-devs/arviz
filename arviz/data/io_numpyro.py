"""NumPyro-specific conversion code."""
import logging
import numpy as np

from .inference_data import InferenceData
from .base import requires, dict_to_dataset
from .. import utils

_log = logging.getLogger(__name__)


class NumPyroConverter:
    """Encapsulate NumPyro specific logic."""

    def __init__(self, *, posterior, prior=None, posterior_predictive=None, coords=None, dims=None):
        """Convert NumPyro data into an InferenceData object.

        Parameters
        ----------
        posterior : numpyro.mcmc.MCMC
            Fitted MCMC object from NumPyro
        prior: dict
            Prior samples from a NumPyro model
        posterior_predictive : dict
            Posterior predictive samples for the posterior
        coords : dict[str] -> list[str]
            Map of dimensions to coordinates
        dims : dict[str] -> list[str]
            Map variable names to their coordinates
        """
        import jax
        import numpyro

        self.posterior = posterior
        self.prior = jax.device_get(prior)
        self.posterior_predictive = jax.device_get(posterior_predictive)
        self.coords = coords
        self.dims = dims
        self.numpyro = numpyro

        posterior_fields = jax.device_get(posterior._samples)  # pylint: disable=protected-access
        # handle the case we run MCMC with a general potential_fn
        # (instead of a NumPyro model) whose args is not a dictionary
        # (e.g. f(x) = x ** 2)
        samples = posterior_fields["z"]
        tree_flatten_samples = jax.tree_util.tree_flatten(samples)[0]
        if not isinstance(samples, dict):
            posterior_fields["z"] = {
                "Param:{}".format(i): jax.device_get(v) for i, v in enumerate(tree_flatten_samples)
            }
        self._posterior_fields = posterior_fields
        self.nchains, self.ndraws = tree_flatten_samples[0].shape[:2]

    @requires("posterior")
    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        data = self._posterior_fields["z"]
        return dict_to_dataset(data, library=self.numpyro, coords=self.coords, dims=self.dims)

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from NumPyro posterior."""
        # match PyMC3 stat names
        rename_key = {
            "potential_energy": "lp",
            "adapt_state.step_size": "step_size",
            "num_steps": "tree_size",
            "accept_prob": "mean_tree_accept",
        }
        data = {}
        for stat, value in self._posterior_fields.items():
            if stat == "z" or not isinstance(value, np.ndarray):
                continue
            name = rename_key.get(stat, stat)
            data[name] = value
            if stat == "num_steps":
                data["depth"] = np.log2(value).astype(int) + 1
        # TODO extract log_likelihood using NumPyro predictive utilities  # pylint: disable=fixme
        return dict_to_dataset(data, library=self.numpyro, coords=self.coords, dims=self.dims)

    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        data = {}
        for k, ary in self.posterior_predictive.items():
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
        return dict_to_dataset(data, library=self.numpyro, coords=self.coords, dims=self.dims)

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        return dict_to_dataset(
            {k: utils.expand_dims(v) for k, v in self.prior.items()},
            library=self.numpyro,
            coords=self.coords,
            dims=self.dims,
        )

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `trace`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        # TODO implement observed_data_to_xarray when model args,  # pylint: disable=fixme
        # kwargs are stored in the next version of NumPyro
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "prior": self.prior_to_xarray(),
            }
        )


def from_numpyro(posterior=None, *, prior=None, posterior_predictive=None, coords=None, dims=None):
    """Convert NumPyro data into an InferenceData object.

    Parameters
    ----------
    posterior : numpyro.mcmc.MCMC
        Fitted MCMC object from NumPyro
    prior: dict
        Prior samples from a NumPyro model
    posterior_predictive : dict
        Posterior predictive samples for the posterior
    coords : dict[str] -> list[str]
        Map of dimensions to coordinates
    dims : dict[str] -> list[str]
        Map variable names to their coordinates
    """
    return NumPyroConverter(
        posterior=posterior,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords=coords,
        dims=dims,
    ).to_inference_data()
