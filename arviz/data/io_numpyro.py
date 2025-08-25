"""NumPyro-specific conversion code."""

from collections import defaultdict
import logging
from typing import Any, Callable, Optional, Dict, List, Tuple

import numpy as np

from .. import utils
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import InferenceData

_log = logging.getLogger(__name__)


def _add_dims(dims_a: Dict[str, List[str]], dims_b: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged = defaultdict(list)

    for k, v in dims_a.items():
        merged[k].extend(v)

    for k, v in dims_b.items():
        merged[k].extend(v)

    # Convert back to a regular dict
    return dict(merged)


def infer_dims(
    model: Callable,
    model_args: Optional[Tuple[Any, ...]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[str]]:

    from numpyro import handlers, distributions as dist
    from numpyro.ops.pytree import PytreeTrace
    from numpyro.infer.initialization import init_to_sample
    import jax

    model_args = tuple() if model_args is None else model_args
    model_kwargs = dict() if model_args is None else model_kwargs

    def _get_dist_name(fn):
        if isinstance(fn, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)):
            return _get_dist_name(fn.base_dist)
        return type(fn).__name__

    def get_trace():
        # We use `init_to_sample` to get around ImproperUniform distribution,
        # which does not have `sample` method.
        subs_model = handlers.substitute(
            handlers.seed(model, 0),
            substitute_fn=init_to_sample,
        )
        trace = handlers.trace(subs_model).get_trace(*model_args, **model_kwargs)
        # Work around an issue where jax.eval_shape does not work
        # for distribution output (e.g. the function `lambda: dist.Normal(0, 1)`)
        # Here we will remove `fn` and store its name in the trace.
        for _, site in trace.items():
            if site["type"] == "sample":
                site["fn_name"] = _get_dist_name(site.pop("fn"))
            elif site["type"] == "deterministic":
                site["fn_name"] = "Deterministic"
        return PytreeTrace(trace)

    # We use eval_shape to avoid any array computation.
    trace = jax.eval_shape(get_trace).trace

    named_dims = {}

    for name, site in trace.items():
        batch_dims = [frame.name for frame in sorted(site["cond_indep_stack"], key=lambda x: x.dim)]
        event_dims = list(site.get("infer", {}).get("event_dims", []))
        if site["type"] in ["sample", "deterministic"] and (batch_dims or event_dims):
            named_dims[name] = batch_dims + event_dims

    return named_dims


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
        extra_event_dims=None,
        num_chains=1,
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
            Map variable names to their coordinates. Will be inferred if they are not provided.
        pred_dims: dict
            Dims for predictions data. Map variable names to their coordinates.
        extra_event_dims: dict
            Extra event dims for deterministic sites. Maps event dims that couldnt be inferred to
            their coordinates.
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
        self.log_likelihood = (
            rcParams["data.log_likelihood"] if log_likelihood is None else log_likelihood
        )
        self.index_origin = rcParams["data.index_origin"] if index_origin is None else index_origin
        self.coords = coords
        self.dims = dims
        self.pred_dims = pred_dims
        self.extra_event_dims = extra_event_dims
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
            self.dims = self.dims if self.dims is not None else self.infer_dims()
            self.pred_dims = (
                self.pred_dims if self.pred_dims is not None else self.infer_pred_dims()
            )
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
            if stat == "potential_energy":
                data[name] = -value
            else:
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

    @requires("posterior")
    @requires("model")
    def infer_dims(self) -> Dict[str, List[str]]:
        dims = infer_dims(self.model, self._args, self._kwargs)
        if self.extra_event_dims:
            dims = _add_dims(dims, self.extra_event_dims)
        return dims

    @requires("posterior")
    @requires("model")
    @requires("predictions")
    def infer_pred_dims(self) -> Dict[str, List[str]]:
        dims = infer_dims(self.model, self._args, self._kwargs)
        if self.extra_event_dims:
            dims = _add_dims(dims, self.extra_event_dims)
        return dims


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
    extra_event_dims=None,
    num_chains=1,
):
    """Convert NumPyro data into an InferenceData object.

    If no dims are provided, this will infer batch dim names from NumPyro model plates.
    For event dim names, such as with the ZeroSumNormal, `infer={"event_dims":dim_names}`
    can be provided in numpyro.sample, i.e.::

        # equivalent to dims entry, {"gamma": ["groups"]}
        gamma = numpyro.sample(
            "gamma",
            dist.ZeroSumNormal(1, event_shape=(n_groups,)),
            infer={"event_dims":["groups"]}
        )

    There is also an additional `extra_event_dims` input to cover any edge cases, for instance
    deterministic sites with event dims (which dont have an `infer` argument to provide metadata).

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
        Map variable names to their coordinates. Will be inferred if they are not provided.
    pred_dims: dict
        Dims for predictions data. Map variable names to their coordinates. Default behavior is to
        infer dims if this is not provided
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
        extra_event_dims=extra_event_dims,
        num_chains=num_chains,
    ).to_inference_data()
