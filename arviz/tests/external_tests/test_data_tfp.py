# pylint: disable=no-member, invalid-name, redefined-outer-name
import numpy as np
import pytest

from ... import from_tfp

from ..helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    importorskip,
    load_cached_models,
)

# Skip all tests if tensorflow_probability not installed
importorskip("tensorflow_probability")


class TestDataTfp:
    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        class Data:
            # Returns result of from_tfp
            model, obj = load_cached_models(
                eight_schools_params, draws, chains, "tensorflow_probability"
            )["tensorflow_probability"]

        return Data

    def get_inference_data(self, data, eight_schools_params):
        """Normal read with observed and var_names."""
        inference_data = from_tfp(
            data.obj,
            var_names=["mu", "tau", "eta"],
            model_fn=lambda: data.model(
                eight_schools_params["J"], eight_schools_params["sigma"].astype(np.float32)
            ),
            observed=eight_schools_params["y"].astype(np.float32),
        )
        return inference_data

    def get_inference_data2(self, data):
        """Fit only."""
        inference_data = from_tfp(data.obj)
        return inference_data

    def get_inference_data3(self, data, eight_schools_params):
        """Read with observed Tensor var_names and dims."""
        import tensorflow as tf

        if int(tf.__version__[0]) > 1:
            import tensorflow.compat.v1 as tf  # pylint: disable=import-error

            tf.disable_v2_behavior()

        inference_data = from_tfp(
            data.obj,
            var_names=["mu", "tau", "eta"],
            model_fn=lambda: data.model(
                eight_schools_params["J"], eight_schools_params["sigma"].astype(np.float32)
            ),
            posterior_predictive_samples=100,
            posterior_predictive_size=3,
            observed=tf.convert_to_tensor(
                np.vstack(
                    (
                        eight_schools_params["y"],
                        eight_schools_params["y"],
                        eight_schools_params["y"],
                    )
                ).astype(np.float32),
                np.float32,
            ),
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"eta": ["school"], "obs": ["size_dim", "school"]},
        )
        return inference_data

    def get_inference_data4(self, data, eight_schools_params):
        """Test setter."""
        inference_data = from_tfp(
            data.obj + [np.ones_like(data.obj[0]).astype(np.float32)],
            var_names=["mu", "tau", "eta", "avg_effect"],
            model_fn=lambda: data.model(
                eight_schools_params["J"], eight_schools_params["sigma"].astype(np.float32)
            ),
            observed=eight_schools_params["y"].astype(np.float32),
        )
        return inference_data

    def test_inference_data(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        test_dict = {
            "posterior": ["mu", "tau", "eta"],
            "observed_data": ["obs"],
            "posterior_predictive": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_inference_data2(self, data):
        inference_data = self.get_inference_data2(data)
        assert hasattr(inference_data, "posterior")

    def test_inference_data3(self, data, eight_schools_params):
        inference_data = self.get_inference_data3(data, eight_schools_params)
        test_dict = {
            "posterior": ["mu", "tau", "eta"],
            "observed_data": ["obs"],
            "posterior_predictive": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_inference_data4(self, data, eight_schools_params):
        inference_data = self.get_inference_data4(data, eight_schools_params)
        test_dict = {
            "posterior": ["mu", "tau", "eta", "avg_effect"],
            "observed_data": ["obs"],
            "posterior_predictive": ["obs"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
