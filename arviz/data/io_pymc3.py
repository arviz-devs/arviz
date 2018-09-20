"""PyMC3-specific conversion code."""
import numpy as np
import xarray as xr

from .inference_data import InferenceData
from .base import requires, dict_to_dataset, generate_dims_coords


class PyMC3Converter:
    """Encapsulate PyMC3 specific logic."""

    def __init__(self, *_, trace=None, prior=None, posterior_predictive=None,
                 coords=None, dims=None):
        self.trace = trace
        self.prior = prior
        self.posterior_predictive = posterior_predictive
        self.coords = coords
        self.dims = dims

    @requires('trace')
    def _extract_log_likelihood(self):
        """Compute log likelihood of each observation.

        Return None if there is not exactly 1 observed random variable.
        """
        # This next line is brittle and may not work forever, but is a secret
        # way to access the model from the trace.
        model = self.trace._straces[0].model  # pylint: disable=protected-access
        if len(model.observed_RVs) != 1:
            return None, None
        else:
            if self.dims is not None:
                coord_name = self.dims.get(model.observed_RVs[0].name)
            else:
                coord_name = None

        cached = [(var, var.logp_elemwise) for var in model.observed_RVs]

        def log_likelihood_vals_point(point):
            """Compute log likelihood for each observed point."""
            log_like_vals = []
            for var, log_like in cached:
                log_like_val = log_like(point)
                if var.missing_values:
                    log_like_val = log_like_val[~var.observations.mask]
                log_like_vals.append(log_like_val.ravel())
            return np.concatenate(log_like_vals)

        chain_likelihoods = []
        for chain in self.trace.chains:
            log_like = (log_likelihood_vals_point(point) for point in self.trace.points([chain]))
            chain_likelihoods.append(np.stack(log_like))
        return np.stack(chain_likelihoods), coord_name

    @requires('trace')
    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        import pymc3 as pm
        var_names = pm.utils.get_default_varnames(self.trace.varnames,  # pylint: disable=no-member
                                                  include_transformed=False)
        data = {}
        for var_name in var_names:
            data[var_name] = np.array(self.trace.get_values(var_name, combine=False,
                                                            squeeze=False))
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('trace')
    def sample_stats_to_xarray(self):
        """Extract sample_stats from PyMC3 trace."""
        rename_key = {
            'model_logp': 'lp',
        }
        data = {}
        for stat in self.trace.stat_names:
            name = rename_key.get(stat, stat)
            data[name] = np.array(self.trace.get_sampler_stats(stat, combine=False))
        log_likelihood, dims = self._extract_log_likelihood()
        if log_likelihood is not None:
            data['log_likelihood'] = log_likelihood
            dims = {'log_likelihood': dims}
        else:
            dims = None

        return dict_to_dataset(data, dims=dims, coords=self.coords)

    @requires('posterior_predictive')
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        data = {k: np.expand_dims(v, 0) for k, v in self.posterior_predictive.items()}
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('prior')
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        return dict_to_dataset({k: np.expand_dims(v, 0) for k, v in self.prior.items()},
                               coords=self.coords,
                               dims=self.dims)

    @requires('trace')
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        # This next line is brittle and may not work forever, but is a secret
        # way to access the model from the trace.
        model = self.trace._straces[0].model  # pylint: disable=protected-access

        observations = {obs.name: obs.observations for obs in model.observed_RVs}
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
        observed_data = {}
        for name, vals in observations.items():
            vals = np.atleast_1d(vals)
            val_dims = dims.get(name)
            val_dims, coords = generate_dims_coords(vals.shape, name,
                                                    dims=val_dims, coords=self.coords)
            # filter coords based on the dims
            coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in val_dims}
            observed_data[name] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=observed_data)

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `trace`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        return InferenceData(**{
            'posterior': self.posterior_to_xarray(),
            'sample_stats': self.sample_stats_to_xarray(),
            'posterior_predictive': self.posterior_predictive_to_xarray(),
            'prior': self.prior_to_xarray(),
            'observed_data': self.observed_data_to_xarray(),
        })


def from_pymc3(*, trace=None, prior=None, posterior_predictive=None,
               coords=None, dims=None):
    """Convert pymc3 data into an InferenceData object."""
    return PyMC3Converter(
        trace=trace,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords=coords,
        dims=dims).to_inference_data()
