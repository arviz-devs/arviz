"""NumPyro-specific conversion code."""
import logging
import numpy as np
import xarray as xr

from .inference_data import InferenceData
from .base import requires, dict_to_dataset, generate_dims_coords, make_attrs
from .. import utils

_log = logging.getLogger(__name__)


class NumPyroConverter:
    """Encapsulate NumPyro specific logic."""

    # pylint: disable=too-many-instance-attributes

    model = None  # type: Optional[callable]
    nchains = None  # type: int
    ndraws = None  # type: int

    def __init__(
        self, *, posterior=None, prior=None, posterior_predictive=None, coords=None, dims=None
    ):
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

        if posterior is not None:
            samples = jax.device_get(self.posterior.get_samples(group_by_chain=True))
            if not isinstance(samples, dict):
                # handle the case we run MCMC with a general potential_fn
                # (instead of a NumPyro model) whose args is not a dictionary
                # (e.g. f(x) = x ** 2)
                tree_flatten_samples = jax.tree_util.tree_flatten(samples)[0]
                samples = {
                    "Param:{}".format(i): jax.device_get(v)
                    for i, v in enumerate(tree_flatten_samples)
                }
            self._samples = samples
            self.nchains, self.ndraws = posterior.num_chains, posterior.num_samples
            self.model = self.posterior.sampler.model
            # model arguments and keyword arguments
            self._args = self.posterior._args  # pylint: disable=protected-access
            self._kwargs = self.posterior._kwargs  # pylint: disable=protected-access
        else:
            self.nchains = self.ndraws = 0

        observations = {}
        if self.model is not None:
            seeded_model = numpyro.handlers.seed(self.model, jax.random.PRNGKey(0))
            trace = numpyro.handlers.trace(seeded_model).get_trace(*self._args, **self._kwargs)
            observations = {
                name: site["value"]
                for name, site in trace.items()
                if site["type"] == "sample" and site["is_observed"]
            }
        self.observations = observations if observations else None

    @requires("posterior")
    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        data = self._samples
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
        for stat, value in self.posterior.get_extra_fields(group_by_chain=True).items():
            if isinstance(value, (dict, tuple)):
                continue
            name = rename_key.get(stat, stat)
            value = value.copy()
            data[name] = value
            if stat == "num_steps":
                data["depth"] = np.log2(value).astype(int) + 1

        # extract log_likelihood
        dims = None
        if self.observations is not None and len(self.observations) == 1:
            samples = self.posterior.get_samples(group_by_chain=False)
            log_likelihood = self.numpyro.infer.log_likelihood(
                self.model, samples, *self._args, **self._kwargs
            )
            obs_name, log_likelihood = list(log_likelihood.items())[0]
            if self.dims is not None:
                coord_name = self.dims.get("log_likelihood", self.dims.get(obs_name))
            else:
                coord_name = None
            shape = (self.nchains, self.ndraws) + log_likelihood.shape[1:]
            data["log_likelihood"] = np.reshape(log_likelihood.copy(), shape)
            dims = {"log_likelihood": coord_name}

        return dict_to_dataset(data, library=self.numpyro, dims=dims, coords=self.coords)

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

    def priors_to_xarray(self):
        """Convert prior samples (and if possible prior predictive too) to xarray."""
        if self.prior is None:
            return {"prior": None, "prior_predictive": None}
        if self.posterior is not None:
            prior_vars = list(self._samples.keys())
            prior_predictive_vars = [key for key in self.prior.keys() if key not in prior_vars]
        else:
            prior_vars = self.prior.keys()
            prior_predictive_vars = None
        priors_dict = {}
        for group, var_names in zip(
            ("prior", "prior_predictive"), (prior_vars, prior_predictive_vars)
        ):
            priors_dict[group] = (
                None
                if var_names is None
                else dict_to_dataset(
                    {k: utils.expand_dims(self.prior[k]) for k in var_names},
                    library=self.numpyro,
                    coords=self.coords,
                    dims=self.dims,
                )
            )
        return priors_dict

    @requires("observations")
    @requires("model")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
        observed_data = {}
        for name, vals in self.observations.items():
            vals = utils.one_de(vals)
            val_dims = dims.get(name)
            val_dims, coords = generate_dims_coords(
                vals.shape, name, dims=val_dims, coords=self.coords
            )
            # filter coords based on the dims
            coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in val_dims}
            observed_data[name] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=observed_data, attrs=make_attrs(library=self.numpyro))

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `trace`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                **self.priors_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
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
