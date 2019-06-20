# pylint: disable=no-member, invalid-name, redefined-outer-name
import pytest
from arviz import from_pyro
from .helpers import (  # pylint: disable=unused-import
    chains,
    draws,
    eight_schools_params,
    load_cached_models,
)


class TestDataPyro:
    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        class Data:
            obj = load_cached_models(eight_schools_params, draws, chains, "pyro")["pyro"]

        return Data

    def get_inference_data(self, data):
        return from_pyro(posterior=data.obj)

    def test_inference_data(self, data):
        inference_data = self.get_inference_data(data)
        assert hasattr(inference_data, "posterior")
