"""Dictionary specific conversion code."""
import warnings
import xarray as xr

from .inference_data import InferenceData
from .base import requires, dict_to_dataset, generate_dims_coords, make_attrs
from .. import utils


class DictConverter:
    """Encapsulate Dictionary specific logic."""

    def __init__(
        self,
        *,
        posterior=None,
        posterior_predictive=None,
        sample_stats=None,
        prior=None,
        prior_predictive=None,
        sample_stats_prior=None,
        observed_data=None,
        coords=None,
        dims=None
    ):
        self.posterior = posterior
        self.posterior_predictive = posterior_predictive
        self.sample_stats = sample_stats
        self.prior = prior
        self.prior_predictive = prior_predictive
        self.sample_stats_prior = sample_stats_prior
        self.observed_data = observed_data
        self.coords = coords
        self.dims = dims

    @requires("posterior")
    def posterior_to_xarray(self):
        """Convert posterior samples to xarray."""
        data = self.posterior
        if not isinstance(data, dict):
            raise TypeError("DictConverter.posterior is not a dictionary")

        if "log_likelihood" in data:
            warnings.warn(
                "log_likelihood found in posterior."
                " For stats functions log_likelihood needs to be in sample_stats.",
                SyntaxWarning,
            )

        return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims)

    @requires("sample_stats")
    def sample_stats_to_xarray(self):
        """Convert sample_stats samples to xarray."""
        data = self.sample_stats
        if not isinstance(data, dict):
            raise TypeError("DictConverter.sample_stats is not a dictionary")

        return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims)

    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        data = self.posterior_predictive
        if not isinstance(data, dict):
            raise TypeError("DictConverter.posterior_predictive is not a dictionary")

        return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims)

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        data = self.prior
        if not isinstance(data, dict):
            raise TypeError("DictConverter.prior is not a dictionary")

        return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims)

    @requires("sample_stats_prior")
    def sample_stats_prior_to_xarray(self):
        """Convert sample_stats_prior samples to xarray."""
        data = self.sample_stats_prior
        if not isinstance(data, dict):
            raise TypeError("DictConverter.sample_stats_prior is not a dictionary")

        return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims)

    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        data = self.prior_predictive
        if not isinstance(data, dict):
            raise TypeError("DictConverter.prior_predictive is not a dictionary")

        return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims)

    @requires("observed_data")
    def observed_data_to_xarray(self):
        """Convert observed_data to xarray."""
        data = self.observed_data
        if not isinstance(data, dict):
            raise TypeError("DictConverter.observed_data is not a dictionary")
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
        observed_data = dict()
        for key, vals in data.items():
            vals = utils.one_de(vals)
            val_dims = dims.get(key)
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            observed_data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=observed_data, attrs=make_attrs(library=None))

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created, then the InferenceData
        will not have those groups.
        """
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "prior": self.prior_to_xarray(),
                "sample_stats_prior": self.sample_stats_prior_to_xarray(),
                "prior_predictive": self.prior_predictive_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
            }
        )


# pylint disable=too-many-instance-attributes
def from_dict(
    posterior=None,
    *,
    posterior_predictive=None,
    sample_stats=None,
    prior=None,
    prior_predictive=None,
    sample_stats_prior=None,
    observed_data=None,
    coords=None,
    dims=None
):
    """Convert Dictionary data into an InferenceData object.

    Parameters
    ----------
    posterior : dict
    posterior_predictive : dict
    sample_stats : dict
        "log_likelihood" variable for stats needs to be here.
    prior : dict
    prior_predictive : dict
    observed_data : dict
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable.

    Returns
    -------
    InferenceData object
    """
    return DictConverter(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        sample_stats=sample_stats,
        prior=prior,
        prior_predictive=prior_predictive,
        sample_stats_prior=sample_stats_prior,
        observed_data=observed_data,
        coords=coords,
        dims=dims,
    ).to_inference_data()
