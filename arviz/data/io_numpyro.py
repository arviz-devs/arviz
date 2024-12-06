"""NumPyro-specific conversion code."""

import logging
from typing import Callable, Optional

import numpy as np

from .. import utils
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import InferenceData

_log = logging.getLogger(__name__)


class NestedToMCMCAdapter:
    """
    Adapter to convert a NestedSampler object into an MCMC-compatible interface.

    This class reshapes posterior samples from a NestedSampler into a chain-and-draw
    structure expected by MCMC workflows, providing compatibility with downstream
    tools like ArviZ for posterior analysis.

    Parameters
    ----------
    nested_sampler : numpyro.contrib.nested_sampling.NestedSampler
        The NestedSampler object containing posterior samples.
    rng_key : jax.random.PRNGKey
        The random key used for sampling.
    num_samples : int
        The total number of posterior samples to draw.
    num_chains : int, optional
        The number of artificial chains to create for MCMC compatibility (default is 1).
    *args : tuple
        Additional positional arguments required by the model (e.g., data, labels).
    **kwargs : dict
        Additional keyword arguments required by the model.

    Attributes
    ----------
    samples : dict
        Reshaped posterior samples organized by variable name.
    thinning : int
        Dummy thinning attribute for compatibility with MCMC.
    sampler : NestedToMCMCAdapter
        Mimics the sampler attribute of an MCMC object.
    model : callable
        The probabilistic model used in the NestedSampler.
    _args : tuple
        Positional arguments passed to the model.
    _kwargs : dict
        Keyword arguments passed to the model.

    Methods
    -------
    get_samples(group_by_chain=True)
        Returns posterior samples reshaped by chain or flattened if `group_by_chain` is False.
    get_extra_fields(group_by_chain=True)
        Provides dummy sampling statistics like accept probabilities, step sizes, and num_steps.
    """

    def __init__(self, nested_sampler, rng_key, num_samples, *args, num_chains=1, **kwargs):
        self.nested_sampler = nested_sampler
        self.rng_key = rng_key
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.samples = self._reshape_samples()
        self.thinning = 1
        self.sampler = self
        self.model = nested_sampler.model
        self._args = args
        self._kwargs = kwargs

    def _reshape_samples(self):
        raw_samples = self.nested_sampler.get_samples(self.rng_key, self.num_samples)
        samples_per_chain = self.num_samples // self.num_chains
        return {
            k: np.reshape(
                v[: samples_per_chain * self.num_chains],
                (self.num_chains, samples_per_chain, *v.shape[1:]),
            )
            for k, v in raw_samples.items()
        }

    def get_samples(self, group_by_chain=True):
        if group_by_chain:
            return self.samples
        else:
            # Flatten chains into a single dimension
            return {k: v.reshape(-1, *v.shape[2:]) for k, v in self.samples.items()}

    def get_extra_fields(self, group_by_chain=True):
        # Generate dummy fields since NestedSampler does not produce these
        n_chains = self.num_chains
        n_samples = self.num_samples // self.num_chains

        # Create dummy values for extra fields
        extra_fields = {
            "accept_prob": np.full((n_chains, n_samples), 1.0),  # Assume all proposals are accepted
            "step_size": np.full((n_chains, n_samples), 0.1),  # Dummy step size
            "num_steps": np.full((n_chains, n_samples), 10),  # Dummy number of steps
        }

        if not group_by_chain:
            # Flatten the chains into a single dimension
            extra_fields = {k: v.reshape(-1, *v.shape[2:]) for k, v in extra_fields.items()}

        return extra_fields


class NumPyroConverter:
    """Encapsulate NumPyro specific logic."""

    # pylint: disable=too-many-instance-attributes

    model = None  # type: Optional[Callable]
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
        log_likelihood=None,
        index_origin=None,
        coords=None,
        dims=None,
        pred_dims=None,
        num_chains=1,
        rng_key=None,
        num_samples=1000,
        data=None,
        labels=None,
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
        index_origin : int, optional
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
        self.rng_key = rng_key
        self.num_samples = num_samples

        if isinstance(posterior, numpyro.contrib.nested_sampling.NestedSampler):
            posterior = NestedToMCMCAdapter(
                posterior, rng_key, num_samples, num_chains=num_chains, data=data, labels=labels
            )
            self.posterior = posterior

        self.prior = jax.device_get(prior)
        self.posterior_predictive = jax.device_get(posterior_predictive)
        self.predictions = predictions
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.log_likelihood = (
            rcParams["data.log_likelihood"] if log_likelihood is None else log_likelihood
        )
        self.index_origin = rcParams["data.index_origin"] if index_origin is None else index_origin
        self.coords = coords
        self.dims = dims
        self.pred_dims = pred_dims
        self.numpyro = numpyro

        def arbitrary_element(dct):
            return next(iter(dct.values()))

        if posterior is not None:
            samples = jax.device_get(self.posterior.get_samples(group_by_chain=True))
            if hasattr(samples, "_asdict"):
                # In case it is easy to convert to a dictionary, as in the case of namedtuples
                samples = samples._asdict()
            if not isinstance(samples, dict):
                # handle the case we run MCMC with a general potential_fn
                # (instead of a NumPyro model) whose args is not a dictionary
                # (e.g. f(x) = x ** 2)
                tree_flatten_samples = jax.tree_util.tree_flatten(samples)[0]
                samples = {
                    f"Param:{i}": jax.device_get(v) for i, v in enumerate(tree_flatten_samples)
                }
            self._samples = samples
            self.nchains, self.ndraws = (
                posterior.num_chains,
                posterior.num_samples // posterior.thinning,
            )
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
            # we need to use an init strategy to generate random samples for ImproperUniform sites
            seeded_model = numpyro.handlers.substitute(
                numpyro.handlers.seed(self.model, jax.random.PRNGKey(0)),
                substitute_fn=numpyro.infer.init_to_sample,
            )
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
        return dict_to_dataset(
            data,
            library=self.numpyro,
            coords=self.coords,
            dims=self.dims,
            index_origin=self.index_origin,
        )

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from NumPyro posterior."""
        rename_key = {
            "potential_energy": "lp",
            "adapt_state.step_size": "step_size",
            "num_steps": "n_steps",
            "accept_prob": "acceptance_rate",
        }
        data = {}
        for stat, value in self.posterior.get_extra_fields(group_by_chain=True).items():
            if isinstance(value, (dict, tuple)):
                continue
            name = rename_key.get(stat, stat)
            value = value.copy()
            data[name] = value
            if stat == "num_steps":
                data["tree_depth"] = np.log2(value).astype(int) + 1
        return dict_to_dataset(
            data,
            library=self.numpyro,
            dims=None,
            coords=self.coords,
            index_origin=self.index_origin,
        )

    @requires("posterior")
    @requires("model")
    def log_likelihood_to_xarray(self):
        """Extract log likelihood from NumPyro posterior."""
        if not self.log_likelihood:
            return None
        data = {}
        if self.observations is not None:
            samples = self.posterior.get_samples(group_by_chain=False)
            if hasattr(samples, "_asdict"):
                samples = samples._asdict()
            log_likelihood_dict = self.numpyro.infer.log_likelihood(
                self.model, samples, *self._args, **self._kwargs
            )
            for obs_name, log_like in log_likelihood_dict.items():
                shape = (self.nchains, self.ndraws) + log_like.shape[1:]
                data[obs_name] = np.reshape(np.asarray(log_like), shape)
        return dict_to_dataset(
            data,
            library=self.numpyro,
            dims=self.dims,
            coords=self.coords,
            index_origin=self.index_origin,
            skip_event_dims=True,
        )

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
        return dict_to_dataset(
            data,
            library=self.numpyro,
            coords=self.coords,
            dims=dims,
            index_origin=self.index_origin,
        )

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
        priors_dict = {
            group: (
                None
                if var_names is None
                else dict_to_dataset(
                    {k: utils.expand_dims(self.prior[k]) for k in var_names},
                    library=self.numpyro,
                    coords=self.coords,
                    dims=self.dims,
                    index_origin=self.index_origin,
                )
            )
            for group, var_names in zip(
                ("prior", "prior_predictive"), (prior_vars, prior_predictive_vars)
            )
        }
        return priors_dict

    @requires("observations")
    @requires("model")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        return dict_to_dataset(
            self.observations,
            library=self.numpyro,
            dims=self.dims,
            coords=self.coords,
            default_dims=[],
            index_origin=self.index_origin,
        )

    @requires("constant_data")
    def constant_data_to_xarray(self):
        """Convert constant_data to xarray."""
        return dict_to_dataset(
            self.constant_data,
            library=self.numpyro,
            dims=self.dims,
            coords=self.coords,
            default_dims=[],
            index_origin=self.index_origin,
        )

    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert predictions_constant_data to xarray."""
        return dict_to_dataset(
            self.predictions_constant_data,
            library=self.numpyro,
            dims=self.pred_dims,
            coords=self.coords,
            default_dims=[],
            index_origin=self.index_origin,
        )

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
    log_likelihood=None,
    index_origin=None,
    coords=None,
    dims=None,
    pred_dims=None,
    num_chains=1,
    rng_key=None,
    num_samples=1000,
    data=None,
    labels=None,
):
    """Convert NumPyro data into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_numpyro <creating_InferenceData>`

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
    index_origin : int, optional
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
        log_likelihood=log_likelihood,
        index_origin=index_origin,
        coords=coords,
        dims=dims,
        pred_dims=pred_dims,
        num_chains=num_chains,
        rng_key=rng_key,
        num_samples=num_samples,
        data=data,
        labels=labels,
    ).to_inference_data()
