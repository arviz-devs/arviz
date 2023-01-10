# pylint: disable=no-member, invalid-name, redefined-outer-name
import numpy as np
import pytest

from ...data.io_beanmachine import from_beanmachine  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
    chains,
    draws,
    eight_schools_params,
    importorskip,
    load_cached_models,
)

# Skip all tests if beanmachine or pytorch not installed
torch = importorskip("torch")
bm = importorskip("beanmachine.ppl")
dist = torch.distributions


class TestDataBeanMachine:
    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        class Data:
            model, prior, obj = load_cached_models(
                eight_schools_params,
                draws,
                chains,
                "beanmachine",
            )["beanmachine"]

        return Data

    @pytest.fixture(scope="class")
    def predictions_data(self, data):
        """Generate predictions for predictions_params"""
        posterior_samples = data.obj
        model = data.model
        predictions = bm.inference.predictive.simulate([model.obs()], posterior_samples)
        return predictions

    def get_inference_data(self, eight_schools_params, predictions_data):
        predictions = predictions_data
        return from_beanmachine(
            sampler=predictions,
            coords={
                "school": np.arange(eight_schools_params["J"]),
                "school_pred": np.arange(eight_schools_params["J"]),
            },
        )

    def test_inference_data(self, data, eight_schools_params, predictions_data):
        inference_data = self.get_inference_data(eight_schools_params, predictions_data)
        model = data.model
        mu = model.mu()
        tau = model.tau()
        eta = model.eta()
        obs = model.obs()

        assert mu in inference_data.posterior
        assert tau in inference_data.posterior
        assert eta in inference_data.posterior
        assert obs in inference_data.posterior_predictive

    def test_inference_data_has_log_likelihood_and_observed_data(self, data):
        idata = from_beanmachine(data.obj)
        obs = data.model.obs()

        assert obs in idata.log_likelihood
        assert obs in idata.observed_data

    def test_inference_data_no_posterior(self, data):
        model = data.model
        # only prior
        inference_data = from_beanmachine(data.prior)
        assert not model.obs() in inference_data.posterior
        assert "observed_data" not in inference_data
