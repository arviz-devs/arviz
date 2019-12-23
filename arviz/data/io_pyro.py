"""Pyro-specific conversion code."""
import logging
import numpy as np
import packaging
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
        if posterior is not None:
            self.nchains, self.ndraws = posterior.num_chains, posterior.num_samples
            if packaging.version.parse(pyro.__version__) >= packaging.version.parse("1.0.0"):
                self.model = self.posterior.kernel.model
                # model arguments and keyword arguments
                self._args = self.posterior._args  # pylint: disable=protected-access
                self._kwargs = self.posterior._kwargs  # pylint: disable=protected-access
        else:
            self.nchains = self.ndraws = 0

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

        # extract log_likelihood
        dims = None
        if self.observations is not None and len(self.observations) == 1:
            obs_name = list(self.observations.keys())[0]
            samples = self.posterior.get_samples(group_by_chain=False)
            predictive = self.pyro.infer.Predictive(self.model, samples)
            obs_site = predictive.get_vectorized_trace(*self._args, **self._kwargs).nodes[obs_name]
            log_likelihood = obs_site["fn"].log_prob(obs_site["value"]).detach().cpu().numpy()
            if self.dims is not None:
                coord_name = self.dims.get("log_likelihood", self.dims.get(obs_name))
            else:
                coord_name = None
            shape = (self.nchains, self.ndraws) + log_likelihood.shape[1:]
            data["log_likelihood"] = np.reshape(log_likelihood, shape)
            dims = {"log_likelihood": coord_name}
        return dict_to_dataset(data, library=self.pyro, coords=self.coords, dims=dims)

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
                    {k: utils.expand_dims(self.prior[k].detach().cpu().numpy()) for k in var_names},
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

    def to_inference_data(self):
        """Convert all available data to an InferenceData object."""
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                **self.priors_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
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
