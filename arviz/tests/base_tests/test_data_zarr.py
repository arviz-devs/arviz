# pylint: disable=redefined-outer-name
import os
import shutil
from collections.abc import MutableMapping

import numpy as np
import pytest

from ... import InferenceData, from_dict

from ..helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    importorskip,
    running_on_ci,
)

zarr = importorskip("zarr")  # pylint: disable=invalid-name


class TestDataZarr:
    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        class Data:
            # fake 8-school output
            obj = {}
            for key, shape in {"mu": [], "tau": [], "eta": [8], "theta": [8]}.items():
                obj[key] = np.random.randn(chains, draws, *shape)

        return Data

    def get_inference_data(self, data, eight_schools_params):
        return from_dict(
            posterior=data.obj,
            posterior_predictive=data.obj,
            sample_stats=data.obj,
            prior=data.obj,
            prior_predictive=data.obj,
            sample_stats_prior=data.obj,
            observed_data=eight_schools_params,
            coords={"school": np.arange(8)},
            dims={"theta": ["school"], "eta": ["school"]},
        )

    @pytest.mark.parametrize("store", [0, 1, 2])
    def test_io_method(self, data, eight_schools_params, store):
        # create InferenceData and check it has been properly created
        inference_data = self.get_inference_data(  # pylint: disable=W0612
            data, eight_schools_params
        )
        test_dict = {
            "posterior": ["eta", "theta", "mu", "tau"],
            "posterior_predictive": ["eta", "theta", "mu", "tau"],
            "sample_stats": ["eta", "theta", "mu", "tau"],
            "prior": ["eta", "theta", "mu", "tau"],
            "prior_predictive": ["eta", "theta", "mu", "tau"],
            "sample_stats_prior": ["eta", "theta", "mu", "tau"],
            "observed_data": ["J", "y", "sigma"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

        # check filename does not exist and use to_zarr method
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "..", "saved_models")
        filepath = os.path.join(data_directory, "zarr")
        assert not os.path.exists(filepath)

        # InferenceData method
        if store == 0:
            # Tempdir
            store = inference_data.to_zarr(store=None)
            assert isinstance(store, MutableMapping)
        elif store == 1:
            inference_data.to_zarr(store=filepath)
            # assert file has been saved correctly
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        elif store == 2:
            store = zarr.storage.DirectoryStore(filepath)
            inference_data.to_zarr(store=store)
            # assert file has been saved correctly
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

        if isinstance(store, MutableMapping):
            inference_data2 = InferenceData.from_zarr(store)
        else:
            inference_data2 = InferenceData.from_zarr(filepath)

        # Everything in dict still available in inference_data2 ?
        fails = check_multiple_attrs(test_dict, inference_data2)
        assert not fails

        # Remove created folder structure
        if os.path.exists(filepath):
            shutil.rmtree(filepath)
        assert not os.path.exists(filepath)
