# pylint: disable=redefined-outer-name
import os
from collections.abc import MutableMapping
from tempfile import TemporaryDirectory
from typing import Mapping

import numpy as np
import pytest

from ... import InferenceData, from_dict
from ... import to_zarr, from_zarr

from ..helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    importorskip,
)

zarr = importorskip("zarr")  # pylint: disable=invalid-name


class TestDataZarr:
    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        class Data:
            # fake 8-school output
            shapes: Mapping[str, list] = {"mu": [], "tau": [], "eta": [8], "theta": [8]}
            obj = {key: np.random.randn(chains, draws, *shape) for key, shape in shapes.items()}

        return Data

    def get_inference_data(self, data, eight_schools_params, fill_attrs):
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
            attrs={"test": 1} if fill_attrs else None,
        )

    @pytest.mark.parametrize("store", [0, 1, 2])
    @pytest.mark.parametrize("fill_attrs", [True, False])
    def test_io_method(self, data, eight_schools_params, store, fill_attrs):
        # create InferenceData and check it has been properly created
        inference_data = self.get_inference_data(  # pylint: disable=W0612
            data, eight_schools_params, fill_attrs
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

        if fill_attrs:
            assert inference_data.attrs["test"] == 1
        else:
            assert "test" not in inference_data.attrs

        # check filename does not exist and use to_zarr method
        with TemporaryDirectory(prefix="arviz_tests_") as tmp_dir:
            filepath = os.path.join(tmp_dir, "zarr")

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

            if fill_attrs:
                assert inference_data2.attrs["test"] == 1
            else:
                assert "test" not in inference_data2.attrs

    def test_io_function(self, data, eight_schools_params):
        # create InferenceData and check it has been properly created
        inference_data = self.get_inference_data(  # pylint: disable=W0612
            data,
            eight_schools_params,
            fill_attrs=True,
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

        assert inference_data.attrs["test"] == 1

        # check filename does not exist and use to_zarr method
        with TemporaryDirectory(prefix="arviz_tests_") as tmp_dir:
            filepath = os.path.join(tmp_dir, "zarr")

            to_zarr(inference_data, store=filepath)
            # assert file has been saved correctly
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

            inference_data2 = from_zarr(filepath)

            # Everything in dict still available in inference_data2 ?
            fails = check_multiple_attrs(test_dict, inference_data2)
            assert not fails

            assert inference_data2.attrs["test"] == 1
