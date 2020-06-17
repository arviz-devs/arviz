"""Dictionary specific conversion code."""
import warnings
import xarray as xr

from .inference_data import InferenceData
from .base import requires, dict_to_dataset, generate_dims_coords, make_attrs
from .. import utils
from ..rcparams import rcParams


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
        warmup_posterior=None,
        warmup_posterior_predictive=None,
        warmup_predictions=None,
        warmup_log_likelihood=None,
        warmup_sample_stats=None,
        save_warmup=None,
        coords=None,
        dims=None,
        pred_dims=None,
        attrs=None,
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
        self.warmup_posterior = warmup_posterior
        self.warmup_posterior_predictive = warmup_posterior_predictive
        self.warmup_predictions = warmup_predictions
        self.warmup_log_likelihood = warmup_log_likelihood
        self.warmup_sample_stats = warmup_sample_stats
        self.save_warmup = rcParams["data.save_warmup"] if save_warmup is None else save_warmup
        self.coords = coords
        self.dims = dims
        self.pred_dims = dims if pred_dims is None else pred_dims
        self.attrs = {} if attrs is None else attrs
        self.attrs.pop("created_at", None)
        self.attrs.pop("arviz_version", None)

    @requires("posterior")
    def posterior_to_xarray(self):
        """Convert posterior samples to xarray."""
        data = self.posterior
        data_warmup = self.warmup_posterior if self.warmup_posterior is not None else {}
        if not isinstance(data, dict):
            raise TypeError("DictConverter.posterior is not a dictionary")
        if not isinstance(data_warmup, dict):
            raise TypeError("DictConverter.warmup_posterior is not a dictionary")

        if "log_likelihood" in data:
            warnings.warn(
                "log_likelihood variable found in posterior group."
                " For stats functions log likelihood data needs to be in log_likelihood group.",
                UserWarning,
            )

        return (
            dict_to_dataset(
                data, library=None, coords=self.coords, dims=self.dims, attrs=self.attrs
            ),
            dict_to_dataset(
                data_warmup, library=None, coords=self.coords, dims=self.dims, attrs=self.attrs
            ),
        )

    @requires("sample_stats")
    def sample_stats_to_xarray(self):
        """Convert sample_stats samples to xarray."""
        data = self.sample_stats
        data_warmup = self.warmup_sample_stats if self.warmup_sample_stats is not None else {}
        if not isinstance(data, dict):
            raise TypeError("DictConverter.sample_stats is not a dictionary")
        if not isinstance(data_warmup, dict):
            raise TypeError("DictConverter.warmup_sample_stats is not a dictionary")

        if "log_likelihood" in data:
            warnings.warn(
                "log_likelihood variable found in sample_stats."
                " Storing log_likelihood data in sample_stats group will be deprecated in "
                "favour of storing them in the log_likelihood group.",
                PendingDeprecationWarning,
            )

        return (
            dict_to_dataset(
                data, library=None, coords=self.coords, dims=self.dims, attrs=self.attrs
            ),
            dict_to_dataset(
                data_warmup, library=None, coords=self.coords, dims=self.dims, attrs=self.attrs
            ),
        )

    @requires("log_likelihood")
    def log_likelihood_to_xarray(self):
        """Convert log_likelihood samples to xarray."""
        data = self.log_likelihood
        data_warmup = self.warmup_log_likelihood if self.warmup_log_likelihood is not None else {}
        if not isinstance(data, dict):
            raise TypeError("DictConverter.log_likelihood is not a dictionary")
        if not isinstance(data_warmup, dict):
            raise TypeError("DictConverter.warmup_log_likelihood is not a dictionary")

        return (
            dict_to_dataset(
                data, library=None, coords=self.coords, dims=self.dims, attrs=self.attrs
            ),
            dict_to_dataset(
                data_warmup, library=None, coords=self.coords, dims=self.dims, attrs=self.attrs
            ),
        )

    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        data = self.posterior_predictive
        data_warmup = (
            self.warmup_posterior_predictive if self.warmup_posterior_predictive is not None else {}
        )
        if not isinstance(data, dict):
            raise TypeError("DictConverter.posterior_predictive is not a dictionary")
        if not isinstance(data_warmup, dict):
            raise TypeError("DictConverter.warmup_posterior_predictive is not a dictionary")

        return (
            dict_to_dataset(
                data, library=None, coords=self.coords, dims=self.dims, attrs=self.attrs
            ),
            dict_to_dataset(
                data_warmup, library=None, coords=self.coords, dims=self.dims, attrs=self.attrs
            ),
        )

    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert predictions to xarray."""
        data = self.predictions
        data_warmup = self.warmup_predictions if self.warmup_predictions is not None else {}
        if not isinstance(data, dict):
            raise TypeError("DictConverter.predictions is not a dictionary")
        if not isinstance(data_warmup, dict):
            raise TypeError("DictConverter.warmup_predictions is not a dictionary")

        return (
            dict_to_dataset(
                data, library=None, coords=self.coords, dims=self.pred_dims, attrs=self.attrs
            ),
            dict_to_dataset(
                data_warmup, library=None, coords=self.coords, dims=self.pred_dims, attrs=self.attrs
            ),
        )

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        data = self.prior
        if not isinstance(data, dict):
            raise TypeError("DictConverter.prior is not a dictionary")

        return dict_to_dataset(
            data, library=None, coords=self.coords, dims=self.dims, attrs=self.attrs
        )

    @requires("sample_stats_prior")
    def sample_stats_prior_to_xarray(self):
        """Convert sample_stats_prior samples to xarray."""
        data = self.sample_stats_prior
        if not isinstance(data, dict):
            raise TypeError("DictConverter.sample_stats_prior is not a dictionary")

        return dict_to_dataset(
            data, library=None, coords=self.coords, dims=self.dims, attrs=self.attrs
        )

    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        data = self.prior_predictive
        if not isinstance(data, dict):
            raise TypeError("DictConverter.prior_predictive is not a dictionary")

        return dict_to_dataset(
            data, library=None, coords=self.coords, dims=self.dims, attrs=self.attrs
        )

    def data_to_xarray(self, dct, group, dims=None):
        """Convert data to xarray."""
        data = dct
        if not isinstance(data, dict):
            raise TypeError("DictConverter.{} is not a dictionary".format(group))
        if dims is None:
            dims = {} if self.dims is None else self.dims
        new_data = dict()
        for key, vals in data.items():
            vals = utils.one_de(vals)
            val_dims = dims.get(key)
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            new_data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=new_data, attrs=make_attrs(attrs=self.attrs, library=None))

    @requires("observed_data")
    def observed_data_to_xarray(self):
        """Convert observed_data to xarray."""
        return self.data_to_xarray(self.observed_data, group="observed_data", dims=self.dims)

    @requires("constant_data")
    def constant_data_to_xarray(self):
        """Convert constant_data to xarray."""
        return self.data_to_xarray(self.constant_data, group="constant_data")

    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert predictions_constant_data to xarray."""
        return self.data_to_xarray(
            self.predictions_constant_data, group="predictions_constant_data", dims=self.pred_dims
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
                "save_warmup": self.save_warmup,
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
    warmup_posterior=None,
    warmup_posterior_predictive=None,
    warmup_predictions=None,
    warmup_log_likelihood=None,
    warmup_sample_stats=None,
    save_warmup=None,
    coords=None,
    dims=None,
    pred_dims=None,
    attrs=None,
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
    warmup_posterior : dict
    warmup_posterior_predictive : dict
    warmup_predictions : dict
    warmup_log_likelihood : dict
    warmup_sample_stats : dict
    save_warmup : bool
        Save warmup iterations InferenceData object. If not defined, use default
        defined by the rcParams.
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable.
    pred_dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for predictions.
    attrs : dict
        A dictionary containing attributes for different groups.

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
        warmup_posterior=warmup_posterior,
        warmup_posterior_predictive=warmup_posterior_predictive,
        warmup_predictions=warmup_predictions,
        warmup_log_likelihood=warmup_log_likelihood,
        warmup_sample_stats=warmup_sample_stats,
        save_warmup=save_warmup,
        coords=coords,
        dims=dims,
        pred_dims=pred_dims,
        attrs=attrs,
    ).to_inference_data()
