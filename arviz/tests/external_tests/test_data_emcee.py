# pylint: disable=no-member, invalid-name, redefined-outer-name
import os

import numpy as np
import pytest

from ... import from_emcee

from ..helpers import _emcee_lnprior as emcee_lnprior
from ..helpers import _emcee_lnprob as emcee_lnprob
from ..helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
    importorskip,
    load_cached_models,
    needs_emcee3_func,
)

# Skip all tests if emcee not installed
emcee = importorskip("emcee")

needs_emcee3 = needs_emcee3_func()


class TestDataEmcee:
    arg_list = [
        ({}, {"posterior": ["var_0", "var_1", "var_7"], "observed_data": ["arg_0", "arg_1"]}),
        (
            {"var_names": ["mu", "tau", "eta"], "slices": [0, 1, slice(2, None)]},
            {
                "posterior": ["mu", "tau", "eta"],
                "observed_data": ["arg_0", "arg_1"],
                "sample_stats": ["lp"],
            },
        ),
        (
            {
                "arg_groups": ["observed_data", "constant_data"],
                "blob_names": ["y", "y"],
                "blob_groups": ["log_likelihood", "posterior_predictive"],
            },
            {
                "posterior": ["var_0", "var_1", "var_7"],
                "observed_data": ["arg_0"],
                "constant_data": ["arg_1"],
                "log_likelihood": ["y"],
                "posterior_predictive": ["y"],
                "sample_stats": ["lp"],
            },
        ),
        (
            {
                "blob_names": ["log_likelihood", "y"],
                "dims": {"eta": ["school"], "log_likelihood": ["school"], "y": ["school"]},
                "var_names": ["mu", "tau", "eta"],
                "slices": [0, 1, slice(2, None)],
                "arg_names": ["y", "sigma"],
                "arg_groups": ["observed_data", "constant_data"],
                "coords": {"school": range(8)},
            },
            {
                "posterior": ["mu", "tau", "eta"],
                "observed_data": ["y"],
                "constant_data": ["sigma"],
                "log_likelihood": ["log_likelihood", "y"],
                "sample_stats": ["lp"],
            },
        ),
    ]

    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        class Data:
            # chains are not used
            # emcee uses lots of walkers
            obj = load_cached_models(eight_schools_params, draws, chains, "emcee")["emcee"]

        return Data

    def get_inference_data_reader(self, **kwargs):
        from emcee import backends  # pylint: disable=no-name-in-module

        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "..", "saved_models")
        filepath = os.path.join(data_directory, "reader_testfile.h5")
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath)
        reader = backends.HDFBackend(filepath, read_only=True)
        return from_emcee(reader, **kwargs)

    @pytest.mark.parametrize("test_args", arg_list)
    def test_inference_data(self, data, test_args):
        kwargs, test_dict = test_args
        inference_data = from_emcee(data.obj, **kwargs)
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    @needs_emcee3
    @pytest.mark.parametrize("test_args", arg_list)
    def test_inference_data_reader(self, test_args):
        kwargs, test_dict = test_args
        kwargs = {k: i for k, i in kwargs.items() if k not in ("arg_names", "arg_groups")}
        inference_data = self.get_inference_data_reader(**kwargs)
        test_dict.pop("observed_data")
        if "constant_data" in test_dict:
            test_dict.pop("constant_data")
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails

    def test_verify_var_names(self, data):
        with pytest.raises(ValueError):
            from_emcee(data.obj, var_names=["not", "enough"])

    def test_verify_arg_names(self, data):
        with pytest.raises(ValueError):
            from_emcee(data.obj, arg_names=["not enough"])

    @pytest.mark.parametrize("slices", [[0, 0, slice(2, None)], [0, 1, slice(1, None)]])
    def test_slices_warning(self, data, slices):
        with pytest.warns(UserWarning):
            from_emcee(data.obj, slices=slices)

    def test_no_blobs_error(self):
        sampler = emcee.EnsembleSampler(6, 1, lambda x: -(x**2))
        sampler.run_mcmc(np.random.normal(size=(6, 1)), 20)
        with pytest.raises(ValueError):
            from_emcee(sampler, blob_names=["inexistent"])

    def test_peculiar_blobs(self, data):
        sampler = emcee.EnsembleSampler(6, 1, lambda x: (-(x**2), (np.random.normal(x), 3)))
        sampler.run_mcmc(np.random.normal(size=(6, 1)), 20)
        inference_data = from_emcee(sampler, blob_names=["normal", "threes"])
        fails = check_multiple_attrs({"log_likelihood": ["normal", "threes"]}, inference_data)
        assert not fails
        inference_data = from_emcee(data.obj, blob_names=["mix"])
        fails = check_multiple_attrs({"log_likelihood": ["mix"]}, inference_data)
        assert not fails

    def test_single_blob(self):
        sampler = emcee.EnsembleSampler(6, 1, lambda x: (-(x**2), 3))
        sampler.run_mcmc(np.random.normal(size=(6, 1)), 20)
        inference_data = from_emcee(sampler, blob_names=["blob"], blob_groups=["blob_group"])
        fails = check_multiple_attrs({"blob_group": ["blob"]}, inference_data)
        assert not fails

    @pytest.mark.parametrize(
        "blob_args",
        [
            (ValueError, ["a", "b"], ["prior"]),
            (ValueError, ["too", "many", "names"], None),
            (SyntaxError, ["a", "b"], ["posterior", "observed_data"]),
        ],
    )
    def test_bad_blobs(self, data, blob_args):
        error, names, groups = blob_args
        with pytest.raises(error):
            from_emcee(data.obj, blob_names=names, blob_groups=groups)

    def test_ln_funcs_for_infinity(self):
        # after dropping Python 3.5 support use underscore 1_000_000
        ary = np.ones(10)
        ary[1] = -1
        assert np.isinf(emcee_lnprior(ary))
        assert np.isinf(emcee_lnprob(ary, ary[2:], ary[2:])[0])
