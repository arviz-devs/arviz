"""Dictionary specific conversion code."""
import warnings
from typing import Optional

from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import WARMUP_TAG, InferenceData


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
        index_origin=None,
        coords=None,
        dims=None,
        pred_dims=None,
        pred_coords=None,
        attrs=None,
        **kwargs,
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
        self.coords = (
            coords
            if pred_coords is None
            else pred_coords
            if coords is None
            else {**coords, **pred_coords}
        )
        self.index_origin = index_origin
        self.coords = coords
        self.dims = dims
        self.pred_dims = dims if pred_dims is None else pred_dims
        self.attrs = {} if attrs is None else attrs
        self.attrs.pop("created_at", None)
        self.attrs.pop("arviz_version", None)
        self._kwargs = kwargs

    def _init_dict(self, attr_name):
        dict_or_none = getattr(self, attr_name, {})
        return {} if dict_or_none is None else dict_or_none

    @requires(["posterior", f"{WARMUP_TAG}posterior"])
    def posterior_to_xarray(self):
        """Convert posterior samples to xarray."""
        data = self._init_dict("posterior")
        data_warmup = self._init_dict(f"{WARMUP_TAG}posterior")
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
        posterior_attrs = self._kwargs.get("posterior_attrs")
        posterior_warmup_attrs = self._kwargs.get("posterior_warmup_attrs")
        return (
            dict_to_dataset(
                data,
                library=None,
                coords=self.coords,
                dims=self.dims,
                attrs=posterior_attrs,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                library=None,
                coords=self.coords,
                dims=self.dims,
                attrs=posterior_warmup_attrs,
                index_origin=self.index_origin,
            ),
        )

    @requires(["sample_stats", f"{WARMUP_TAG}sample_stats"])
    def sample_stats_to_xarray(self):
        """Convert sample_stats samples to xarray."""
        data = self._init_dict("sample_stats")
        data_warmup = self._init_dict(f"{WARMUP_TAG}sample_stats")
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
        sample_stats_attrs = self._kwargs.get("sample_stats_attrs")
        sample_stats_warmup_attrs = self._kwargs.get("sample_stats_warmup_attrs")
        return (
            dict_to_dataset(
                data,
                library=None,
                coords=self.coords,
                dims=self.dims,
                attrs=sample_stats_attrs,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                library=None,
                coords=self.coords,
                dims=self.dims,
                attrs=sample_stats_warmup_attrs,
                index_origin=self.index_origin,
            ),
        )

    @requires(["log_likelihood", f"{WARMUP_TAG}log_likelihood"])
    def log_likelihood_to_xarray(self):
        """Convert log_likelihood samples to xarray."""
        data = self._init_dict("log_likelihood")
        data_warmup = self._init_dict(f"{WARMUP_TAG}log_likelihood")
        if not isinstance(data, dict):
            raise TypeError("DictConverter.log_likelihood is not a dictionary")
        if not isinstance(data_warmup, dict):
            raise TypeError("DictConverter.warmup_log_likelihood is not a dictionary")
        log_likelihood_attrs = self._kwargs.get("log_likelihood_attrs")
        log_likelihood_warmup_attrs = self._kwargs.get("log_likelihood_warmup_attrs")
        return (
            dict_to_dataset(
                data,
                library=None,
                coords=self.coords,
                dims=self.dims,
                attrs=log_likelihood_attrs,
                index_origin=self.index_origin,
                skip_event_dims=True,
            ),
            dict_to_dataset(
                data_warmup,
                library=None,
                coords=self.coords,
                dims=self.dims,
                attrs=log_likelihood_warmup_attrs,
                index_origin=self.index_origin,
                skip_event_dims=True,
            ),
        )

    @requires(["posterior_predictive", f"{WARMUP_TAG}posterior_predictive"])
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        data = self._init_dict("posterior_predictive")
        data_warmup = self._init_dict(f"{WARMUP_TAG}posterior_predictive")
        if not isinstance(data, dict):
            raise TypeError("DictConverter.posterior_predictive is not a dictionary")
        if not isinstance(data_warmup, dict):
            raise TypeError("DictConverter.warmup_posterior_predictive is not a dictionary")
        posterior_predictive_attrs = self._kwargs.get("posterior_predictive_attrs")
        posterior_predictive_warmup_attrs = self._kwargs.get("posterior_predictive_warmup_attrs")
        return (
            dict_to_dataset(
                data,
                library=None,
                coords=self.coords,
                dims=self.dims,
                attrs=posterior_predictive_attrs,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                library=None,
                coords=self.coords,
                dims=self.dims,
                attrs=posterior_predictive_warmup_attrs,
                index_origin=self.index_origin,
            ),
        )

    @requires(["predictions", f"{WARMUP_TAG}predictions"])
    def predictions_to_xarray(self):
        """Convert predictions to xarray."""
        data = self._init_dict("predictions")
        data_warmup = self._init_dict(f"{WARMUP_TAG}predictions")
        if not isinstance(data, dict):
            raise TypeError("DictConverter.predictions is not a dictionary")
        if not isinstance(data_warmup, dict):
            raise TypeError("DictConverter.warmup_predictions is not a dictionary")
        predictions_attrs = self._kwargs.get("predictions_attrs")
        predictions_warmup_attrs = self._kwargs.get("predictions_warmup_attrs")
        return (
            dict_to_dataset(
                data,
                library=None,
                coords=self.coords,
                dims=self.pred_dims,
                attrs=predictions_attrs,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                library=None,
                coords=self.coords,
                dims=self.pred_dims,
                attrs=predictions_warmup_attrs,
                index_origin=self.index_origin,
            ),
        )

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        data = self.prior
        if not isinstance(data, dict):
            raise TypeError("DictConverter.prior is not a dictionary")
        prior_attrs = self._kwargs.get("prior_attrs")
        return dict_to_dataset(
            data,
            library=None,
            coords=self.coords,
            dims=self.dims,
            attrs=prior_attrs,
            index_origin=self.index_origin,
        )

    @requires("sample_stats_prior")
    def sample_stats_prior_to_xarray(self):
        """Convert sample_stats_prior samples to xarray."""
        data = self.sample_stats_prior
        if not isinstance(data, dict):
            raise TypeError("DictConverter.sample_stats_prior is not a dictionary")
        sample_stats_prior_attrs = self._kwargs.get("sample_stats_prior_attrs")
        return dict_to_dataset(
            data,
            library=None,
            coords=self.coords,
            dims=self.dims,
            attrs=sample_stats_prior_attrs,
            index_origin=self.index_origin,
        )

    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        data = self.prior_predictive
        if not isinstance(data, dict):
            raise TypeError("DictConverter.prior_predictive is not a dictionary")
        prior_predictive_attrs = self._kwargs.get("prior_predictive_attrs")
        return dict_to_dataset(
            data,
            library=None,
            coords=self.coords,
            dims=self.dims,
            attrs=prior_predictive_attrs,
            index_origin=self.index_origin,
        )

    def data_to_xarray(self, data, group, dims=None):
        """Convert data to xarray."""
        if not isinstance(data, dict):
            raise TypeError(f"DictConverter.{group} is not a dictionary")
        if dims is None:
            dims = {} if self.dims is None else self.dims
        return dict_to_dataset(
            data,
            library=None,
            coords=self.coords,
            dims=self.dims,
            default_dims=[],
            attrs=self.attrs,
            index_origin=self.index_origin,
        )

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
                "attrs": self.attrs,
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
    index_origin: Optional[int] = None,
    coords=None,
    dims=None,
    pred_dims=None,
    pred_coords=None,
    attrs=None,
    **kwargs,
):
    """Convert Dictionary data into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_dict <creating_InferenceData>`

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
    index_origin : int, optional
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable.
    pred_dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for predictions.
    pred_coords : dict[str, List(str)]
        A mapping from variables to a list of coordinate values for predictions.
    attrs : dict
        A dictionary containing attributes for different groups.
    kwargs : dict
        A dictionary containing group attrs.
        Accepted kwargs are:
        - posterior_attrs, posterior_warmup_attrs : attrs for posterior group
        - sample_stats_attrs, sample_stats_warmup_attrs : attrs for sample_stats group
        - log_likelihood_attrs, log_likelihood_warmup_attrs : attrs for log_likelihood group
        - posterior_predictive_attrs, posterior_predictive_warmup_attrs : attrs for
                posterior_predictive group
        - predictions_attrs, predictions_warmup_attrs : attrs for predictions group
        - prior_attrs : attrs for prior group
        - sample_stats_prior_attrs : attrs for sample_stats_prior group
        - prior_predictive_attrs : attrs for prior_predictive group

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
        index_origin=index_origin,
        coords=coords,
        dims=dims,
        pred_dims=pred_dims,
        pred_coords=pred_coords,
        attrs=attrs,
        **kwargs,
    ).to_inference_data()
