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
        self,
        *,
        posterior=None,
        prior=None,
        posterior_predictive=None,
        predictions=None,
        constant_data=None,
        predictions_constant_data=None,
        coords=None,
        dims=None,
        pred_dims=None,
        num_chains=1
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
        predictions: dict
            Out of sample predictions
        constant_data: dict
            Dictionary containing constant data variables mapped to their values.
        predictions_constant_data: dict
            Constant data used for out-of-sample predictions.
        coords : dict[str] -> list[str]
            Map of dimensions to coordinates
        dims : dict[str] -> list[str]
            Map variable names to their coordinates
        pred_dims: dict
            Dims for predictions data. Map variable names to their coordinates.
        num_chains: int
            Number of chains used for sampling. Ignored if posterior is present.
        """
        import jax
        import numpyro

        self.posterior = posterior
        self.prior = jax.device_get(prior)
        self.posterior_predictive = jax.device_get(posterior_predictive)
        self.predictions = predictions
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.coords = coords
        self.dims = dims
        self.pred_dims = pred_dims
        self.numpyro = numpyro

        def arbitrary_element(dct):
            return next(iter(dct.values()))

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
            self.nchains = num_chains
            get_from = None
            if predictions is not None:
                get_from = predictions
            elif posterior_predictive is not None:
                get_from = posterior_predictive
            elif prior is not None:
                get_from = prior
            if get_from is None and constant_data is None and predictions_constant_data is None:
                raise ValueError(
                    "When constructing InferenceData must have at least"
                    " one of posterior, prior, posterior_predictive or predictions."
                )
            if get_from is not None:
                aelem = arbitrary_element(get_from)
                self.ndraws = aelem.shape[0] // self.nchains

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
        return dict_to_dataset(data, library=self.numpyro, dims=None, coords=self.coords)

    @requires("posterior")
    @requires("model")
    def log_likelihood_to_xarray(self):
        """Extract log likelihood from NumPyro posterior."""
        data = {}
        if self.observations is not None:
            samples = self.posterior.get_samples(group_by_chain=False)
            log_likelihood_dict = self.numpyro.infer.log_likelihood(
                self.model, samples, *self._args, **self._kwargs
            )
            for obs_name, log_like in log_likelihood_dict.items():
                shape = (self.nchains, self.ndraws) + log_like.shape[1:]
                data[obs_name] = np.reshape(log_like.copy(), shape)
        return dict_to_dataset(data, library=self.numpyro, dims=self.dims, coords=self.coords)

    def translate_posterior_predictive_dict_to_xarray(self, dct, dims):
        """Convert posterior_predictive or prediction samples to xarray."""
        data = {}
        for k, ary in dct.items():
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
        return dict_to_dataset(data, library=self.numpyro, coords=self.coords, dims=dims)

    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        return self.translate_posterior_predictive_dict_to_xarray(
            self.posterior_predictive, self.dims
        )

    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert predictions to xarray."""
        return self.translate_posterior_predictive_dict_to_xarray(self.predictions, self.pred_dims)

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

    def convert_constant_data_to_xarray(self, dct, dims):
        """Convert constant_data or predictions_constant_data to xarray."""
        if dims is None:
            dims = {}
        constant_data = {}
        for name, vals in dct.items():
            vals = utils.one_de(vals)
            val_dims = dims.get(name)
            val_dims, coords = generate_dims_coords(
                vals.shape, name, dims=val_dims, coords=self.coords
            )
            # filter coords based on the dims
            coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in val_dims}
            constant_data[name] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=constant_data, attrs=make_attrs(library=self.numpyro))

    @requires("constant_data")
    def constant_data_to_xarray(self):
        """Convert constant_data to xarray."""
        return self.convert_constant_data_to_xarray(self.constant_data, self.dims)

    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert predictions_constant_data to xarray."""
        return self.convert_constant_data_to_xarray(self.predictions_constant_data, self.pred_dims)

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
                "log_likelihood": self.log_likelihood_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "predictions": self.predictions_to_xarray(),
                **self.priors_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
                "constant_data": self.constant_data_to_xarray(),
                "predictions_constant_data": self.predictions_constant_data_to_xarray(),
            }
        )


def from_numpyro(
    posterior=None,
    *,
    prior=None,
    posterior_predictive=None,
    predictions=None,
    constant_data=None,
    predictions_constant_data=None,
    coords=None,
    dims=None,
    pred_dims=None,
    num_chains=1
):
    """Convert NumPyro data into an InferenceData object.

    For a usage example read the
    :doc:`Cookbook section on from_numpyro </notebooks/InferenceDataCookbook>`

    Parameters
    ----------
    posterior : numpyro.mcmc.MCMC
        Fitted MCMC object from NumPyro
    prior: dict
        Prior samples from a NumPyro model
    posterior_predictive : dict
        Posterior predictive samples for the posterior
    predictions: dict
        Out of sample predictions
    constant_data: dict
        Dictionary containing constant data variables mapped to their values.
    predictions_constant_data: dict
        Constant data used for out-of-sample predictions.
    coords : dict[str] -> list[str]
        Map of dimensions to coordinates
    dims : dict[str] -> list[str]
        Map variable names to their coordinates
    pred_dims: dict
        Dims for predictions data. Map variable names to their coordinates.
    num_chains: int
        Number of chains used for sampling. Ignored if posterior is present.
    """
    return NumPyroConverter(
        posterior=posterior,
        prior=prior,
        posterior_predictive=posterior_predictive,
        predictions=predictions,
        constant_data=constant_data,
        predictions_constant_data=predictions_constant_data,
        coords=coords,
        dims=dims,
        pred_dims=pred_dims,
        num_chains=num_chains,
    ).to_inference_data()
