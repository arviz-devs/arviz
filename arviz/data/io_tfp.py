"""Tfp-specific conversion code."""
import numpy as np
import xarray as xr

from .. import utils
from .base import dict_to_dataset, generate_dims_coords, make_attrs
from .inference_data import InferenceData


# pylint: disable=too-many-instance-attributes
class TfpConverter:
    """Encapsulate tfp specific logic."""

    def __init__(
        self,
        *,
        posterior,
        var_names=None,
        model_fn=None,
        feed_dict=None,
        posterior_predictive_samples=100,
        posterior_predictive_size=1,
        chain_dim=None,
        observed=None,
        coords=None,
        dims=None,
    ):

        self.posterior = posterior

        if var_names is None:
            self.var_names = []
            for i in range(0, len(posterior)):
                self.var_names.append(f"var_{i}")
        else:
            self.var_names = var_names

        self.model_fn = model_fn
        self.feed_dict = feed_dict
        self.posterior_predictive_samples = posterior_predictive_samples
        self.posterior_predictive_size = posterior_predictive_size
        self.observed = observed
        self.chain_dim = chain_dim
        self.coords = coords
        self.dims = dims

        import tensorflow as tf
        import tensorflow_probability as tfp
        import tensorflow_probability.python.edward2 as ed

        self.tfp = tfp
        self.tf = tf  # pylint: disable=invalid-name
        self.ed = ed  # pylint: disable=invalid-name

        if int(self.tf.__version__[0]) > 1:
            import tensorflow.compat.v1 as tf  # pylint: disable=import-error

            tf.disable_v2_behavior()
            self.tf = tf  # pylint: disable=invalid-name

    def handle_chain_location(self, ary):
        """Move the axis corresponding to the chain to first position.

        If there is only one chain which has no axis, add it.
        """
        if self.chain_dim is None:
            return utils.expand_dims(ary)
        return ary.swapaxes(0, self.chain_dim)

    def posterior_to_xarray(self):
        """Convert the posterior to an xarray dataset."""
        data = {}
        for i, var_name in enumerate(self.var_names):
            data[var_name] = self.handle_chain_location(self.posterior[i])
        return dict_to_dataset(data, library=self.tfp, coords=self.coords, dims=self.dims)

    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        if self.observed is None:
            return None

        observed_data = {}
        if isinstance(self.observed, self.tf.Tensor):
            with self.tf.Session() as sess:
                vals = sess.run(self.observed, feed_dict=self.feed_dict)
        else:
            vals = self.observed

        if self.dims is None:
            dims = {}
        else:
            dims = self.dims

        name = "obs"
        val_dims = dims.get(name)
        vals = utils.one_de(vals)
        val_dims, coords = generate_dims_coords(vals.shape, name, dims=val_dims, coords=self.coords)
        # coords = {key: xr.IndexVariable((key,), data=coords[key]) for key in val_dims}

        observed_data[name] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=observed_data, attrs=make_attrs(library=self.tfp))

    def _value_setter(self, variables):
        def interceptor(rv_constructor, *rv_args, **rv_kwargs):
            """Replace prior on effects with empirical posterior mean from MCMC."""
            name = rv_kwargs.pop("name")
            if name in variables:
                rv_kwargs["value"] = variables[name]
            return rv_constructor(*rv_args, **rv_kwargs)

        return interceptor

    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        if self.model_fn is None:
            return None

        posterior_preds = []
        sample_size = self.posterior[0].shape[0]

        for i in np.arange(0, sample_size, int(sample_size / self.posterior_predictive_samples)):
            variables = {}
            for var_i, var_name in enumerate(self.var_names):
                variables[var_name] = self.posterior[var_i][i]

            with self.ed.interception(self._value_setter(variables)):
                if self.posterior_predictive_size > 1:
                    posterior_preds.append(
                        [self.model_fn() for _ in range(self.posterior_predictive_size)]
                    )
                else:
                    posterior_preds.append(self.model_fn())

        data = {}
        with self.tf.Session() as sess:
            data["obs"] = self.handle_chain_location(
                sess.run(posterior_preds, feed_dict=self.feed_dict)
            )
        return dict_to_dataset(data, library=self.tfp, coords=self.coords, dims=self.dims)

    def sample_stats_to_xarray(self):
        """Extract sample_stats from tfp trace."""
        if self.model_fn is None or self.observed is None:
            return None

        log_likelihood = []
        sample_size = self.posterior[0].shape[0]

        for i in range(sample_size):
            variables = {}
            for var_i, var_name in enumerate(self.var_names):
                variables[var_name] = self.posterior[var_i][i]

            with self.ed.interception(self._value_setter(variables)):
                log_likelihood.append((self.model_fn().distribution.log_prob(self.observed)))

        data = {}
        if self.dims is not None:
            coord_name = self.dims.get("obs")
        else:
            coord_name = None
        dims = {"log_likelihood": coord_name}

        with self.tf.Session() as sess:
            data["log_likelihood"] = self.handle_chain_location(
                sess.run(log_likelihood, feed_dict=self.feed_dict)
            )
        return dict_to_dataset(data, library=self.tfp, coords=self.coords, dims=dims)

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
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
            }
        )


def from_tfp(
    posterior=None,
    *,
    var_names=None,
    model_fn=None,
    feed_dict=None,
    posterior_predictive_samples=100,
    posterior_predictive_size=1,
    chain_dim=None,
    observed=None,
    coords=None,
    dims=None,
):
    """Convert tfp data into an InferenceData object."""
    return TfpConverter(
        posterior=posterior,
        var_names=var_names,
        model_fn=model_fn,
        feed_dict=feed_dict,
        posterior_predictive_samples=posterior_predictive_samples,
        posterior_predictive_size=posterior_predictive_size,
        chain_dim=chain_dim,
        observed=observed,
        coords=coords,
        dims=dims,
    ).to_inference_data()
