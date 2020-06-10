"""Pyro-specific conversion code."""
import logging
import warnings
import numpy as np
from packaging import version
import xarray as xr

from .inference_data import InferenceData
from .base import requires, dict_to_dataset, generate_dims_coords, make_attrs
from .. import utils

_log = logging.getLogger(__name__)


class PyroConverter:
    """Encapsulate Pyro specific logic."""

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
        log_likelihood=True,
        predictions=None,
        constant_data=None,
        predictions_constant_data=None,
        coords=None,
        dims=None,
        pred_dims=None,
        num_chains=1,
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
        self.posterior = posterior
        self.prior = prior
        self.posterior_predictive = posterior_predictive
        self.log_likelihood = log_likelihood
        self.predictions = predictions
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.coords = coords
        self.dims = dims
        self.pred_dims = pred_dims
        import pyro

        def arbitrary_element(dct):
            return next(iter(dct.values()))

        self.pyro = pyro
        if posterior is not None:
            self.nchains, self.ndraws = posterior.num_chains, posterior.num_samples
            if version.parse(pyro.__version__) >= version.parse("1.0.0"):
                self.model = self.posterior.kernel.model
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
            trace = pyro.poutine.trace(self.model).get_trace(*self._args, **self._kwargs)
            observations = {
                name: site["value"]
                for name, site in trace.nodes.items()
                if site["type"] == "sample" and site["is_observed"]
            }
        self.observations = observations if observations else None

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
        return dict_to_dataset(data, library=self.pyro, coords=self.coords, dims=None)

    @requires("posterior")
    @requires("model")
    def log_likelihood_to_xarray(self):
        """Extract log likelihood from Pyro posterior."""
        if not self.log_likelihood:
            return None
        data = {}
        if self.observations is not None:
            try:
                samples = self.posterior.get_samples(group_by_chain=False)
                predictive = self.pyro.infer.Predictive(self.model, samples)
                vectorized_trace = predictive.get_vectorized_trace(*self._args, **self._kwargs)
                for obs_name in self.observations.keys():
                    obs_site = vectorized_trace.nodes[obs_name]
                    log_like = obs_site["fn"].log_prob(obs_site["value"]).detach().cpu().numpy()
                    shape = (self.nchains, self.ndraws) + log_like.shape[1:]
                    data[obs_name] = np.reshape(log_like, shape)
            except:  # pylint: disable=bare-except
                # cannot get vectorized trace
                warnings.warn(
                    "Could not get vectorized trace, log_likelihood group will be omitted. "
                    "Check your model vectorization or set log_likelihood=False"
                )
                return None
        return dict_to_dataset(data, library=self.pyro, coords=self.coords, dims=self.dims)

    def translate_posterior_predictive_dict_to_xarray(self, dct, dims):
        """Convert posterior_predictive or prediction samples to xarray."""
        data = {}
        for k, ary in dct.items():
            ary = ary.detach().cpu().numpy()
            shape = ary.shape
            if shape[0] == self.nchains and shape[1] == self.ndraws:
                data[k] = ary
            elif shape[0] == self.nchains * self.ndraws:
                data[k] = ary.reshape((self.nchains, self.ndraws, *shape[1:]))
            else:
                data[k] = utils.expand_dims(ary)
                _log.warning(
                    "posterior predictive shape not compatible with number of chains and draws."
                    "This can mean that some draws or even whole chains are not represented."
                )
        return dict_to_dataset(data, library=self.pyro, coords=self.coords, dims=dims)

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
            prior_vars = list(self.posterior.get_samples().keys())
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
                    {
                        k: utils.expand_dims(np.squeeze(self.prior[k].detach().cpu().numpy()))
                        for k in var_names
                    },
                    library=self.pyro,
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
        return xr.Dataset(data_vars=observed_data, attrs=make_attrs(library=self.pyro))

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
        return xr.Dataset(data_vars=constant_data, attrs=make_attrs(library=self.pyro))

    @requires("constant_data")
    def constant_data_to_xarray(self):
        """Convert constant_data to xarray."""
        return self.convert_constant_data_to_xarray(self.constant_data, self.dims)

    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert predictions_constant_data to xarray."""
        return self.convert_constant_data_to_xarray(self.predictions_constant_data, self.pred_dims)

    def to_inference_data(self):
        """Convert all available data to an InferenceData object."""
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                "log_likelihood": self.log_likelihood_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "predictions": self.predictions_to_xarray(),
                "constant_data": self.constant_data_to_xarray(),
                "predictions_constant_data": self.predictions_constant_data_to_xarray(),
                **self.priors_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
            }
        )


def from_pyro(
    posterior=None,
    *,
    prior=None,
    posterior_predictive=None,
    log_likelihood=True,
    predictions=None,
    constant_data=None,
    predictions_constant_data=None,
    coords=None,
    dims=None,
    pred_dims=None,
    num_chains=1,
):
    """Convert Pyro data into an InferenceData object.

    For a usage example read the
    :doc:`Cookbook section on from_pyro </notebooks/InferenceDataCookbook>`

    Parameters
    ----------
    posterior : pyro.infer.MCMC
        Fitted MCMC object from Pyro
    prior: dict
        Prior samples from a Pyro model
    posterior_predictive : dict
        Posterior predictive samples for the posterior
    log_likelihood : bool, optional
        Calculate and store pointwise log likelihood values.
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
    return PyroConverter(
        posterior=posterior,
        prior=prior,
        posterior_predictive=posterior_predictive,
        log_likelihood=log_likelihood,
        predictions=predictions,
        constant_data=constant_data,
        predictions_constant_data=predictions_constant_data,
        coords=coords,
        dims=dims,
        pred_dims=pred_dims,
        num_chains=num_chains,
    ).to_inference_data()
