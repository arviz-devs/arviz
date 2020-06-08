"""Dictionary specific conversion code."""
import warnings
import xarray as xr

from .inference_data import InferenceData
from .base import requires, dict_to_dataset, generate_dims_coords, make_attrs
from .. import utils


# pylint: disable=too-many-instance-attributes
class DictConverter:
    """Encapsulate Dictionary specific logic."""

    def __init__(
        self,
        *,
        posterior=None,
        posterior_predictive=None,
        predictions=None,
        sample_stats=None,
        log_likelihood=None,
        prior=None,
        prior_predictive=None,
        sample_stats_prior=None,
        observed_data=None,
        constant_data=None,
        predictions_constant_data=None,
        coords=None,
        dims=None
    ):
        self.posterior = posterior
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions
        self.sample_stats = sample_stats
        self.log_likelihood = log_likelihood
        self.prior = prior
        self.prior_predictive = prior_predictive
        self.sample_stats_prior = sample_stats_prior
        self.observed_data = observed_data
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
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
                "log_likelihood variable found in posterior group."
                " For stats functions log likelihood data needs to be in log_likelihood group.",
                UserWarning,
            )

        return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims)

    @requires("sample_stats")
    def sample_stats_to_xarray(self):
        """Convert sample_stats samples to xarray."""
        data = self.sample_stats
        if not isinstance(data, dict):
            raise TypeError("DictConverter.sample_stats is not a dictionary")

        if "log_likelihood" in data:
            warnings.warn(
                "log_likelihood variable found in sample_stats."
                " Storing log_likelihood data in sample_stats group will be deprecated in "
                "favour of storing them in the log_likelihood group.",
                PendingDeprecationWarning,
            )

        return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims)

    @requires("log_likelihood")
    def log_likelihood_to_xarray(self):
        """Convert log_likelihood samples to xarray."""
        data = self.log_likelihood
        if not isinstance(data, dict):
            raise TypeError("DictConverter.log_likelihood is not a dictionary")

        return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims)

    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        data = self.posterior_predictive
        if not isinstance(data, dict):
            raise TypeError("DictConverter.posterior_predictive is not a dictionary")

        return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims)

    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert predictions to xarray."""
        data = self.predictions
        if not isinstance(data, dict):
            raise TypeError("DictConverter.predictions is not a dictionary")

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

    def data_to_xarray(self, dct, group):
        """Convert data to xarray."""
        data = dct
        if not isinstance(data, dict):
            raise TypeError("DictConverter.{} is not a dictionary".format(group))
        if self.dims is None:
            dims = {}
        else:
            dims = self.dims
        new_data = dict()
        for key, vals in data.items():
            vals = utils.one_de(vals)
            val_dims = dims.get(key)
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            new_data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=new_data, attrs=make_attrs(library=None))

    @requires("observed_data")
    def observed_data_to_xarray(self):
        """Convert observed_data to xarray."""
        return self.data_to_xarray(self.observed_data, group="observed_data")

    @requires("constant_data")
    def constant_data_to_xarray(self):
        """Convert constant_data to xarray."""
        return self.data_to_xarray(self.constant_data, group="constant_data")

    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert predictions_constant_data to xarray."""
        return self.data_to_xarray(
            self.predictions_constant_data, group="predictions_constant_data"
        )

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created, then the InferenceData
        will not have those groups.
        """
        return InferenceData(
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                "log_likelihood": self.log_likelihood_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "predictions": self.predictions_to_xarray(),
                "prior": self.prior_to_xarray(),
                "sample_stats_prior": self.sample_stats_prior_to_xarray(),
                "prior_predictive": self.prior_predictive_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
                "constant_data": self.constant_data_to_xarray(),
                "predictions_constant_data": self.predictions_constant_data_to_xarray(),
            }
        )


# pylint: disable=too-many-instance-attributes
def from_dict(
    posterior=None,
    *,
    posterior_predictive=None,
    predictions=None,
    sample_stats=None,
    log_likelihood=None,
    prior=None,
    prior_predictive=None,
    sample_stats_prior=None,
    observed_data=None,
    constant_data=None,
    predictions_constant_data=None,
    coords=None,
    dims=None
):
    """Convert Dictionary data into an InferenceData object.

    For a usage example read the
    :doc:`Cookbook section on from_dict </notebooks/InferenceDataCookbook>`

    Parameters
    ----------
    posterior : dict
    posterior_predictive : dict
    predictions: dict
    sample_stats : dict
    log_likelihood : dict
        For stats functions, log likelihood data should be stored here.
    prior : dict
    prior_predictive : dict
    observed_data : dict
    constant_data : dict
    predictions_constant_data: dict
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
        predictions=predictions,
        sample_stats=sample_stats,
        log_likelihood=log_likelihood,
        prior=prior,
        prior_predictive=prior_predictive,
        sample_stats_prior=sample_stats_prior,
        observed_data=observed_data,
        constant_data=constant_data,
        predictions_constant_data=predictions_constant_data,
        coords=coords,
        dims=dims,
    ).to_inference_data()
