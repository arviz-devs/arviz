"""beanmachine-specific conversion code."""

from .inference_data import InferenceData
from .base import dict_to_dataset, requires


class BMConverter:
    """Encapsulate Bean Machine specific logic."""

    def __init__(
        self,
        *,
        sampler=None,
        coords=None,
        dims=None,
    ) -> None:
        self.sampler = sampler
        self.coords = coords
        self.dims = dims

        import beanmachine.ppl as bm

        self.beanm = bm

        if "posterior" in self.sampler.namespaces:
            self.posterior = self.sampler.namespaces["posterior"].samples
        else:
            self.posterior = None

        if "posterior_predictive" in self.sampler.namespaces:
            self.posterior_predictive = self.sampler.namespaces["posterior_predictive"].samples
        else:
            self.posterior_predictive = None

        if self.sampler.log_likelihoods is not None:
            self.log_likelihoods = self.sampler.log_likelihoods
        else:
            self.log_likelihoods = None

        if self.sampler.observations is not None:
            self.observations = self.sampler.observations
        else:
            self.observations = None

    @requires("posterior")
    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        data = {k: v.detach().cpu().numpy() for k, v in self.posterior.items()}
        return dict_to_dataset(data, library=self.beanm, coords=self.coords, dims=self.dims)

    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        data = {k: v.detach().cpu().numpy() for k, v in self.posterior_predictive.items()}
        return dict_to_dataset(data, library=self.beanm, coords=self.coords, dims=self.dims)

    @requires("log_likelihoods")
    def log_likelihood_to_xarray(self):
        data = {k: v.detach().cpu().numpy() for k, v in self.log_likelihoods.items()}
        return dict_to_dataset(data, library=self.beanm, coords=self.coords, dims=self.dims)

    @requires("observations")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        data = {k: v.detach().cpu().numpy() for k, v in self.observations.items()}
        return dict_to_dataset(
            data, library=self.beanm, coords=self.coords, dims=self.dims, default_dims=[]
        )

    def to_inference_data(self):
        """Convert all available data to an InferenceData object."""
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "log_likelihood": self.log_likelihood_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
            }
        )


def from_beanmachine(
    sampler=None,
    *,
    coords=None,
    dims=None,
):
    """Convert Bean Machine MonteCarloSamples object into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_beanmachine <creating_InferenceData>`


    Parameters
    ----------
    sampler : bm.MonteCarloSamples
        Fitted MonteCarloSamples object from Bean Machine
    coords : dict of {str : array-like}
        Map of dimensions to coordinates
    dims : dict of {str : list of str}
        Map variable names to their coordinates
    """
    return BMConverter(
        sampler=sampler,
        coords=coords,
        dims=dims,
    ).to_inference_data()
