# pylint: disable=no-member, invalid-name, redefined-outer-name
# pylint: disable=too-many-lines

import importlib
import os
import warnings
from collections import namedtuple
from copy import deepcopy
from html import escape
from typing import Dict
from tempfile import TemporaryDirectory
from urllib.parse import urlunsplit

import numpy as np
import pytest
import xarray as xr
from xarray.core.options import OPTIONS
from xarray.testing import assert_identical

from ... import (
    InferenceData,
    clear_data_home,
    concat,
    convert_to_dataset,
    convert_to_inference_data,
    from_datatree,
    from_dict,
    from_json,
    from_netcdf,
    list_datasets,
    load_arviz_data,
    to_netcdf,
    extract,
)

from ...data.base import (
    dict_to_dataset,
    generate_dims_coords,
    infer_stan_dtypes,
    make_attrs,
    numpy_to_data_array,
)
from ...data.datasets import LOCAL_DATASETS, REMOTE_DATASETS, RemoteFileMetadata
from ..helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    create_data_random,
    data_random,
    draws,
    eight_schools_params,
    models,
)

# Check if dm-tree is installed
dm_tree_installed = importlib.util.find_spec("tree") is not None  # pylint: disable=invalid-name
skip_tests = (not dm_tree_installed) and ("ARVIZ_REQUIRE_ALL_DEPS" not in os.environ)


@pytest.fixture(autouse=True)
def no_remote_data(monkeypatch, tmpdir):
    """Delete all remote data and replace it with a local dataset."""
    keys = list(REMOTE_DATASETS)
    for key in keys:
        monkeypatch.delitem(REMOTE_DATASETS, key)

    centered = LOCAL_DATASETS["centered_eight"]
    filename = os.path.join(str(tmpdir), os.path.basename(centered.filename))

    url = urlunsplit(("file", "", centered.filename, "", ""))

    monkeypatch.setitem(
        REMOTE_DATASETS,
        "test_remote",
        RemoteFileMetadata(
            name="test_remote",
            filename=filename,
            url=url,
            checksum="8efc3abafe0c796eb9aea7b69490d4e2400a33c57504ef4932e1c7105849176f",
            description=centered.description,
        ),
    )
    monkeypatch.setitem(
        REMOTE_DATASETS,
        "bad_checksum",
        RemoteFileMetadata(
            name="bad_checksum",
            filename=filename,
            url=url,
            checksum="bad!",
            description=centered.description,
        ),
    )
    UnknownFileMetaData = namedtuple(
        "UnknownFileMetaData", ["filename", "url", "checksum", "description"]
    )
    monkeypatch.setitem(
        REMOTE_DATASETS,
        "test_unknown",
        UnknownFileMetaData(
            filename=filename,
            url=url,
            checksum="9ae00c83654b3f061d32c882ec0a270d10838fa36515ecb162b89a290e014849",
            description="Test bad REMOTE_DATASET",
        ),
    )


def test_load_local_arviz_data():
    inference_data = load_arviz_data("centered_eight")
    assert isinstance(inference_data, InferenceData)
    assert set(inference_data.observed_data.obs.coords["school"].values) == {
        "Hotchkiss",
        "Mt. Hermon",
        "Choate",
        "Deerfield",
        "Phillips Andover",
        "St. Paul's",
        "Lawrenceville",
        "Phillips Exeter",
    }
    assert inference_data.posterior["theta"].dims == ("chain", "draw", "school")


@pytest.mark.parametrize("fill_attrs", [True, False])
def test_local_save(fill_attrs):
    inference_data = load_arviz_data("centered_eight")
    assert isinstance(inference_data, InferenceData)

    if fill_attrs:
        inference_data.attrs["test"] = 1
    with TemporaryDirectory(prefix="arviz_tests_") as tmp_dir:
        path = os.path.join(tmp_dir, "test_file.nc")
        inference_data.to_netcdf(path)

        inference_data2 = from_netcdf(path)
        if fill_attrs:
            assert "test" in inference_data2.attrs
            assert inference_data2.attrs["test"] == 1
        # pylint: disable=protected-access
        assert all(group in inference_data2 for group in inference_data._groups_all)
        # pylint: enable=protected-access


def test_clear_data_home():
    resource = REMOTE_DATASETS["test_remote"]
    assert not os.path.exists(resource.filename)
    load_arviz_data("test_remote")
    assert os.path.exists(resource.filename)
    clear_data_home(data_home=os.path.dirname(resource.filename))
    assert not os.path.exists(resource.filename)


def test_load_remote_arviz_data():
    assert load_arviz_data("test_remote")


def test_bad_checksum():
    with pytest.raises(IOError):
        load_arviz_data("bad_checksum")


def test_missing_dataset():
    with pytest.raises(ValueError):
        load_arviz_data("does not exist")


def test_list_datasets():
    dataset_string = list_datasets()
    # make sure all the names of the data sets are in the dataset description
    for key in (
        "centered_eight",
        "non_centered_eight",
        "test_remote",
        "bad_checksum",
        "test_unknown",
    ):
        assert key in dataset_string


def test_dims_coords():
    shape = 4, 20, 5
    var_name = "x"
    dims, coords = generate_dims_coords(shape, var_name)
    assert "x_dim_0" in dims
    assert "x_dim_1" in dims
    assert "x_dim_2" in dims
    assert len(coords["x_dim_0"]) == 4
    assert len(coords["x_dim_1"]) == 20
    assert len(coords["x_dim_2"]) == 5


@pytest.mark.parametrize(
    "in_dims", (["dim1", "dim2"], ["draw", "dim1", "dim2"], ["chain", "draw", "dim1", "dim2"])
)
def test_dims_coords_default_dims(in_dims):
    shape = 4, 7
    var_name = "x"
    dims, coords = generate_dims_coords(
        shape,
        var_name,
        dims=in_dims,
        coords={"chain": ["a", "b", "c"]},
        default_dims=["chain", "draw"],
    )
    assert "dim1" in dims
    assert "dim2" in dims
    assert ("chain" in dims) == ("chain" in in_dims)
    assert ("draw" in dims) == ("draw" in in_dims)
    assert len(coords["dim1"]) == 4
    assert len(coords["dim2"]) == 7
    assert len(coords["chain"]) == 3
    assert "draw" not in coords


def test_dims_coords_extra_dims():
    shape = 4, 20
    var_name = "x"
    with pytest.warns(UserWarning):
        dims, coords = generate_dims_coords(shape, var_name, dims=["xx", "xy", "xz"])
    assert "xx" in dims
    assert "xy" in dims
    assert "xz" in dims
    assert len(coords["xx"]) == 4
    assert len(coords["xy"]) == 20


@pytest.mark.parametrize("shape", [(4, 20), (4, 20, 1)])
def test_dims_coords_skip_event_dims(shape):
    coords = {"x": np.arange(4), "y": np.arange(20), "z": np.arange(5)}
    dims, coords = generate_dims_coords(
        shape, "name", dims=["x", "y", "z"], coords=coords, skip_event_dims=True
    )
    assert "x" in dims
    assert "y" in dims
    assert "z" not in dims
    assert len(coords["x"]) == 4
    assert len(coords["y"]) == 20
    assert "z" not in coords


@pytest.mark.parametrize("dims", [None, ["chain", "draw"], ["chain", "draw", None]])
def test_numpy_to_data_array_with_dims(dims):
    da = numpy_to_data_array(
        np.empty((4, 500, 7)),
        var_name="a",
        dims=dims,
        default_dims=["chain", "draw"],
    )
    assert list(da.dims) == ["chain", "draw", "a_dim_0"]


def test_make_attrs():
    extra_attrs = {"key": "Value"}
    attrs = make_attrs(attrs=extra_attrs)
    assert "key" in attrs
    assert attrs["key"] == "Value"


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("sequence", [True, False])
def test_concat_group(copy, inplace, sequence):
    idata1 = from_dict(
        posterior={"A": np.random.randn(2, 10, 2), "B": np.random.randn(2, 10, 5, 2)}
    )
    if copy and inplace:
        original_idata1_posterior_id = id(idata1.posterior)
    idata2 = from_dict(prior={"C": np.random.randn(2, 10, 2), "D": np.random.randn(2, 10, 5, 2)})
    idata3 = from_dict(observed_data={"E": np.random.randn(100), "F": np.random.randn(2, 100)})
    # basic case
    assert concat(idata1, idata2, copy=True, inplace=False) is not None
    if sequence:
        new_idata = concat((idata1, idata2, idata3), copy=copy, inplace=inplace)
    else:
        new_idata = concat(idata1, idata2, idata3, copy=copy, inplace=inplace)
    if inplace:
        assert new_idata is None
        new_idata = idata1
    assert new_idata is not None
    test_dict = {"posterior": ["A", "B"], "prior": ["C", "D"], "observed_data": ["E", "F"]}
    fails = check_multiple_attrs(test_dict, new_idata)
    assert not fails
    if copy:
        if inplace:
            assert id(new_idata.posterior) == original_idata1_posterior_id
        else:
            assert id(new_idata.posterior) != id(idata1.posterior)
        assert id(new_idata.prior) != id(idata2.prior)
        assert id(new_idata.observed_data) != id(idata3.observed_data)
    else:
        assert id(new_idata.posterior) == id(idata1.posterior)
        assert id(new_idata.prior) == id(idata2.prior)
        assert id(new_idata.observed_data) == id(idata3.observed_data)


@pytest.mark.parametrize("dim", ["chain", "draw"])
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("sequence", [True, False])
@pytest.mark.parametrize("reset_dim", [True, False])
def test_concat_dim(dim, copy, inplace, sequence, reset_dim):
    idata1 = from_dict(
        posterior={"A": np.random.randn(2, 10, 2), "B": np.random.randn(2, 10, 5, 2)},
        observed_data={"C": np.random.randn(100), "D": np.random.randn(2, 100)},
    )
    if inplace:
        original_idata1_id = id(idata1)
    idata2 = from_dict(
        posterior={"A": np.random.randn(2, 10, 2), "B": np.random.randn(2, 10, 5, 2)},
        observed_data={"C": np.random.randn(100), "D": np.random.randn(2, 100)},
    )
    idata3 = from_dict(
        posterior={"A": np.random.randn(2, 10, 2), "B": np.random.randn(2, 10, 5, 2)},
        observed_data={"C": np.random.randn(100), "D": np.random.randn(2, 100)},
    )
    # basic case
    assert (
        concat(idata1, idata2, dim=dim, copy=copy, inplace=False, reset_dim=reset_dim) is not None
    )
    if sequence:
        new_idata = concat(
            (idata1, idata2, idata3), copy=copy, dim=dim, inplace=inplace, reset_dim=reset_dim
        )
    else:
        new_idata = concat(
            idata1, idata2, idata3, dim=dim, copy=copy, inplace=inplace, reset_dim=reset_dim
        )
    if inplace:
        assert new_idata is None
        new_idata = idata1
    assert new_idata is not None
    test_dict = {"posterior": ["A", "B"], "observed_data": ["C", "D"]}
    fails = check_multiple_attrs(test_dict, new_idata)
    assert not fails
    if inplace:
        assert id(new_idata) == original_idata1_id
    else:
        assert id(new_idata) != id(idata1)
    assert getattr(new_idata.posterior, dim).size == 6 if dim == "chain" else 30
    if reset_dim:
        assert np.all(
            getattr(new_idata.posterior, dim).values
            == (np.arange(6) if dim == "chain" else np.arange(30))
        )


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("sequence", [True, False])
def test_concat_edgecases(copy, inplace, sequence):
    idata = from_dict(posterior={"A": np.random.randn(2, 10, 2), "B": np.random.randn(2, 10, 5, 2)})
    empty = concat()
    assert empty is not None
    if sequence:
        new_idata = concat([idata], copy=copy, inplace=inplace)
    else:
        new_idata = concat(idata, copy=copy, inplace=inplace)
    if inplace:
        assert new_idata is None
        new_idata = idata
    else:
        assert new_idata is not None
    test_dict = {"posterior": ["A", "B"]}
    fails = check_multiple_attrs(test_dict, new_idata)
    assert not fails
    if copy and not inplace:
        assert id(new_idata.posterior) != id(idata.posterior)
    else:
        assert id(new_idata.posterior) == id(idata.posterior)


def test_concat_bad():
    with pytest.raises(TypeError):
        concat("hello", "hello")
    idata = from_dict(posterior={"A": np.random.randn(2, 10, 2), "B": np.random.randn(2, 10, 5, 2)})
    idata2 = from_dict(posterior={"A": np.random.randn(2, 10, 2)})
    idata3 = from_dict(prior={"A": np.random.randn(2, 10, 2)})
    with pytest.raises(TypeError):
        concat(idata, np.array([1, 2, 3, 4, 5]))
    with pytest.raises(TypeError):
        concat(idata, idata, dim=None)
    with pytest.raises(TypeError):
        concat(idata, idata2, dim="chain")
    with pytest.raises(TypeError):
        concat(idata2, idata, dim="chain")
    with pytest.raises(TypeError):
        concat(idata, idata3, dim="chain")
    with pytest.raises(TypeError):
        concat(idata3, idata, dim="chain")


def test_inference_concat_keeps_all_fields():
    """From failures observed in issue #907"""
    idata1 = from_dict(posterior={"A": [1, 2, 3, 4]}, sample_stats={"B": [2, 3, 4, 5]})
    idata2 = from_dict(prior={"C": [1, 2, 3, 4]}, observed_data={"D": [2, 3, 4, 5]})

    idata_c1 = concat(idata1, idata2)
    idata_c2 = concat(idata2, idata1)

    test_dict = {"posterior": ["A"], "sample_stats": ["B"], "prior": ["C"], "observed_data": ["D"]}

    fails_c1 = check_multiple_attrs(test_dict, idata_c1)
    assert not fails_c1
    fails_c2 = check_multiple_attrs(test_dict, idata_c2)
    assert not fails_c2


@pytest.mark.parametrize(
    "model_code,expected",
    [
        ("data {int y;} models {y ~ poisson(3);} generated quantities {int X;}", {"X": "int"}),
        (
            "data {real y;} models {y ~ normal(0,1);} generated quantities {int Y; real G;}",
            {"Y": "int"},
        ),
    ],
)
def test_infer_stan_dtypes(model_code, expected):
    """Test different examples for dtypes in Stan models."""
    res = infer_stan_dtypes(model_code)
    assert res == expected


class TestInferenceData:  # pylint: disable=too-many-public-methods
    def test_addition(self):
        idata1 = from_dict(
            posterior={"A": np.random.randn(2, 10, 2), "B": np.random.randn(2, 10, 5, 2)}
        )
        idata2 = from_dict(
            prior={"C": np.random.randn(2, 10, 2), "D": np.random.randn(2, 10, 5, 2)}
        )
        new_idata = idata1 + idata2
        assert new_idata is not None
        test_dict = {"posterior": ["A", "B"], "prior": ["C", "D"]}
        fails = check_multiple_attrs(test_dict, new_idata)
        assert not fails

    def test_iter(self, models):
        idata = models.model_1
        for group in idata:
            assert group in idata._groups_all  # pylint: disable=protected-access

    def test_groups(self, models):
        idata = models.model_1
        for group in idata.groups():
            assert group in idata._groups_all  # pylint: disable=protected-access

    def test_values(self, models):
        idata = models.model_1
        datasets = idata.values()
        for group in idata.groups():
            assert group in idata._groups_all  # pylint: disable=protected-access
            dataset = getattr(idata, group)
            assert dataset in datasets

    def test_items(self, models):
        idata = models.model_1
        for group, dataset in idata.items():
            assert group in idata._groups_all  # pylint: disable=protected-access
            assert dataset.equals(getattr(idata, group))

    @pytest.mark.parametrize("inplace", [True, False])
    def test_extend_xr_method(self, data_random, inplace):
        idata = data_random
        idata_copy = deepcopy(idata)
        kwargs = {"groups": "posterior_groups"}
        if inplace:
            idata_copy.sum(dim="draw", inplace=inplace, **kwargs)
        else:
            idata2 = idata_copy.sum(dim="draw", inplace=inplace, **kwargs)
            assert idata2 is not idata_copy
            idata_copy = idata2
        assert_identical(idata_copy.posterior, idata.posterior.sum(dim="draw"))
        assert_identical(
            idata_copy.posterior_predictive, idata.posterior_predictive.sum(dim="draw")
        )
        assert_identical(idata_copy.observed_data, idata.observed_data)

    @pytest.mark.parametrize("inplace", [False, True])
    def test_sel(self, data_random, inplace):
        idata = data_random
        original_groups = getattr(idata, "_groups")
        ndraws = idata.posterior.draw.values.size
        kwargs = {"draw": slice(200, None), "chain": slice(None, None, 2), "b_dim_0": [1, 2, 7]}
        if inplace:
            idata.sel(inplace=inplace, **kwargs)
        else:
            idata2 = idata.sel(inplace=inplace, **kwargs)
            assert idata2 is not idata
            idata = idata2
        groups = getattr(idata, "_groups")
        assert np.all(np.isin(groups, original_groups))
        for group in groups:
            dataset = getattr(idata, group)
            assert "b_dim_0" in dataset.dims
            assert np.all(dataset.b_dim_0.values == np.array(kwargs["b_dim_0"]))
            if group != "observed_data":
                assert np.all(np.isin(["chain", "draw"], dataset.dims))
                assert np.all(dataset.chain.values == np.arange(0, 4, 2))
                assert np.all(dataset.draw.values == np.arange(200, ndraws))

    def test_sel_chain_prior(self):
        idata = load_arviz_data("centered_eight")
        original_groups = getattr(idata, "_groups")
        idata_subset = idata.sel(inplace=False, chain_prior=False, chain=[0, 1, 3])
        groups = getattr(idata_subset, "_groups")
        assert np.all(np.isin(groups, original_groups))
        for group in groups:
            dataset_subset = getattr(idata_subset, group)
            dataset = getattr(idata, group)
            if "chain" in dataset.dims:
                assert "chain" in dataset_subset.dims
                if "prior" not in group:
                    assert np.all(dataset_subset.chain.values == np.array([0, 1, 3]))
            else:
                assert "chain" not in dataset_subset.dims
        with pytest.raises(KeyError):
            idata.sel(inplace=False, chain_prior=True, chain=[0, 1, 3])

    @pytest.mark.parametrize("use", ("del", "delattr", "delitem"))
    def test_del(self, use):
        # create inference data object
        data = np.random.normal(size=(4, 500, 8))
        idata = from_dict(
            posterior={"a": data[..., 0], "b": data},
            sample_stats={"a": data[..., 0], "b": data},
            observed_data={"b": data[0, 0, :]},
            posterior_predictive={"a": data[..., 0], "b": data},
        )

        # assert inference data object has all attributes
        test_dict = {
            "posterior": ("a", "b"),
            "sample_stats": ("a", "b"),
            "observed_data": ["b"],
            "posterior_predictive": ("a", "b"),
        }
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails
        # assert _groups attribute contains all groups
        groups = getattr(idata, "_groups")
        assert all((group in groups for group in test_dict))

        # Use del method
        if use == "del":
            del idata.sample_stats
        elif use == "delitem":
            del idata["sample_stats"]
        else:
            delattr(idata, "sample_stats")

        # assert attribute has been removed
        test_dict.pop("sample_stats")
        fails = check_multiple_attrs(test_dict, idata)
        assert not fails
        assert not hasattr(idata, "sample_stats")
        # assert _groups attribute has been updated
        assert "sample_stats" not in getattr(idata, "_groups")

    @pytest.mark.parametrize(
        "args_res",
        (
            ([("posterior", "sample_stats")], ("posterior", "sample_stats")),
            (["posterior", "like"], ("posterior", "warmup_posterior", "posterior_predictive")),
            (["^posterior", "regex"], ("posterior", "posterior_predictive")),
            (
                [("~^warmup", "~^obs"), "regex"],
                ("posterior", "sample_stats", "posterior_predictive"),
            ),
            (
                ["~observed_vars"],
                ("posterior", "sample_stats", "warmup_posterior", "warmup_sample_stats"),
            ),
        ),
    )
    def test_group_names(self, args_res):
        args, result = args_res
        ds = dict_to_dataset({"a": np.random.normal(size=(3, 10))})
        idata = InferenceData(
            posterior=(ds, ds),
            sample_stats=(ds, ds),
            observed_data=ds,
            posterior_predictive=ds,
        )
        group_names = idata._group_names(*args)  # pylint: disable=protected-access
        assert np.all([name in result for name in group_names])

    def test_group_names_invalid_args(self):
        ds = dict_to_dataset({"a": np.random.normal(size=(3, 10))})
        idata = InferenceData(posterior=(ds, ds))
        msg = r"^\'filter_groups\' can only be None, \'like\', or \'regex\', got: 'foo'$"
        with pytest.raises(ValueError, match=msg):
            idata._group_names(  # pylint: disable=protected-access
                ("posterior",), filter_groups="foo"
            )

    @pytest.mark.parametrize("inplace", [False, True])
    def test_isel(self, data_random, inplace):
        idata = data_random
        original_groups = getattr(idata, "_groups")
        ndraws = idata.posterior.draw.values.size
        kwargs = {"draw": slice(200, None), "chain": slice(None, None, 2), "b_dim_0": [1, 2, 7]}
        if inplace:
            idata.isel(inplace=inplace, **kwargs)
        else:
            idata2 = idata.isel(inplace=inplace, **kwargs)
            assert idata2 is not idata
            idata = idata2
        groups = getattr(idata, "_groups")
        assert np.all(np.isin(groups, original_groups))
        for group in groups:
            dataset = getattr(idata, group)
            assert "b_dim_0" in dataset.dims
            assert np.all(dataset.b_dim_0.values == np.array(kwargs["b_dim_0"]))
            if group != "observed_data":
                assert np.all(np.isin(["chain", "draw"], dataset.dims))
                assert np.all(dataset.chain.values == np.arange(0, 4, 2))
                assert np.all(dataset.draw.values == np.arange(200, ndraws))

    def test_rename(self, data_random):
        idata = data_random
        original_groups = getattr(idata, "_groups")
        renamed_idata = idata.rename({"b": "b_new"})
        for group in original_groups:
            xr_data = getattr(renamed_idata, group)
            assert "b_new" in list(xr_data.data_vars)
            assert "b" not in list(xr_data.data_vars)

        renamed_idata = idata.rename({"b_dim_0": "b_new"})
        for group in original_groups:
            xr_data = getattr(renamed_idata, group)
            assert "b_new" in list(xr_data.dims)
            assert "b_dim_0" not in list(xr_data.dims)

    def test_rename_vars(self, data_random):
        idata = data_random
        original_groups = getattr(idata, "_groups")
        renamed_idata = idata.rename_vars({"b": "b_new"})
        for group in original_groups:
            xr_data = getattr(renamed_idata, group)
            assert "b_new" in list(xr_data.data_vars)
            assert "b" not in list(xr_data.data_vars)

        renamed_idata = idata.rename_vars({"b_dim_0": "b_new"})
        for group in original_groups:
            xr_data = getattr(renamed_idata, group)
            assert "b_new" not in list(xr_data.dims)
            assert "b_dim_0" in list(xr_data.dims)

    def test_rename_dims(self, data_random):
        idata = data_random
        original_groups = getattr(idata, "_groups")
        renamed_idata = idata.rename_dims({"b_dim_0": "b_new"})
        for group in original_groups:
            xr_data = getattr(renamed_idata, group)
            assert "b_new" in list(xr_data.dims)
            assert "b_dim_0" not in list(xr_data.dims)

        renamed_idata = idata.rename_dims({"b": "b_new"})
        for group in original_groups:
            xr_data = getattr(renamed_idata, group)
            assert "b_new" not in list(xr_data.data_vars)
            assert "b" in list(xr_data.data_vars)

    def test_stack_unstack(self):
        datadict = {
            "a": np.random.randn(100),
            "b": np.random.randn(1, 100, 10),
            "c": np.random.randn(1, 100, 3, 4),
        }
        coords = {
            "c1": np.arange(3),
            "c99": np.arange(4),
            "b1": np.arange(10),
        }
        dims = {"c": ["c1", "c99"], "b": ["b1"]}
        dataset = from_dict(posterior=datadict, coords=coords, dims=dims)
        assert_identical(
            dataset.stack(z=["c1", "c99"]).posterior, dataset.posterior.stack(z=["c1", "c99"])
        )
        assert_identical(dataset.stack(z=["c1", "c99"]).unstack().posterior, dataset.posterior)
        assert_identical(
            dataset.stack(z=["c1", "c99"]).unstack(dim="z").posterior, dataset.posterior
        )

    def test_stack_bool(self):
        datadict = {
            "a": np.random.randn(100),
            "b": np.random.randn(1, 100, 10),
            "c": np.random.randn(1, 100, 3, 4),
        }
        coords = {
            "c1": np.arange(3),
            "c99": np.arange(4),
            "b1": np.arange(10),
        }
        dims = {"c": ["c1", "c99"], "b": ["b1"]}
        dataset = from_dict(posterior=datadict, coords=coords, dims=dims)
        assert_identical(
            dataset.stack(z=["c1", "c99"], create_index=False).posterior,
            dataset.posterior.stack(z=["c1", "c99"], create_index=False),
        )

    def test_to_dict(self, models):
        idata = models.model_1
        test_data = from_dict(**idata.to_dict())
        assert test_data
        for group in idata._groups_all:  # pylint: disable=protected-access
            xr_data = getattr(idata, group)
            test_xr_data = getattr(test_data, group)
            assert xr_data.equals(test_xr_data)

    def test_to_dict_warmup(self):
        idata = create_data_random(
            groups=[
                "posterior",
                "sample_stats",
                "observed_data",
                "warmup_posterior",
                "warmup_posterior_predictive",
            ]
        )
        test_data = from_dict(**idata.to_dict(), save_warmup=True)
        assert test_data
        for group in idata._groups_all:  # pylint: disable=protected-access
            xr_data = getattr(idata, group)
            test_xr_data = getattr(test_data, group)
            assert xr_data.equals(test_xr_data)

    @pytest.mark.parametrize(
        "kwargs",
        (
            {
                "groups": "posterior",
                "include_coords": True,
                "include_index": True,
                "index_origin": 0,
            },
            {
                "groups": ["posterior", "sample_stats"],
                "include_coords": False,
                "include_index": True,
                "index_origin": 0,
            },
            {
                "groups": "posterior_groups",
                "include_coords": True,
                "include_index": False,
                "index_origin": 1,
            },
        ),
    )
    def test_to_dataframe(self, kwargs):
        idata = from_dict(
            posterior={"a": np.random.randn(4, 100, 3, 4, 5), "b": np.random.randn(4, 100)},
            sample_stats={"a": np.random.randn(4, 100, 3, 4, 5), "b": np.random.randn(4, 100)},
            observed_data={"a": np.random.randn(3, 4, 5), "b": np.random.randn(4)},
        )
        test_data = idata.to_dataframe(**kwargs)
        assert not test_data.empty
        groups = kwargs.get("groups", idata._groups_all)  # pylint: disable=protected-access
        for group in idata._groups_all:  # pylint: disable=protected-access
            if "data" in group:
                continue
            assert test_data.shape == (
                (4 * 100, 3 * 4 * 5 + 1 + 2)
                if groups == "posterior"
                else (4 * 100, (3 * 4 * 5 + 1) * 2 + 2)
            )
            if groups == "posterior":
                if kwargs.get("include_coords", True) and kwargs.get("include_index", True):
                    assert any(
                        f"[{kwargs.get('index_origin', 0)}," in item[0]
                        for item in test_data.columns
                        if isinstance(item, tuple)
                    )
                if kwargs.get("include_coords", True):
                    assert any(isinstance(item, tuple) for item in test_data.columns)
                else:
                    assert not any(isinstance(item, tuple) for item in test_data.columns)
            else:
                if not kwargs.get("include_index", True):
                    assert all(
                        item in test_data.columns
                        for item in (("posterior", "a", 1, 1, 1), ("posterior", "b"))
                    )
            assert all(item in test_data.columns for item in ("chain", "draw"))

    @pytest.mark.parametrize(
        "kwargs",
        (
            {
                "var_names": ["parameter_1", "parameter_2", "variable_1", "variable_2"],
                "filter_vars": None,
                "var_results": [
                    ("posterior", "parameter_1"),
                    ("posterior", "parameter_2"),
                    ("prior", "parameter_1"),
                    ("prior", "parameter_2"),
                    ("posterior", "variable_1"),
                    ("posterior", "variable_2"),
                ],
            },
            {
                "var_names": "parameter",
                "filter_vars": "like",
                "groups": "posterior",
                "var_results": ["parameter_1", "parameter_2"],
            },
            {
                "var_names": "~parameter",
                "filter_vars": "like",
                "groups": "posterior",
                "var_results": ["variable_1", "variable_2", "custom_name"],
            },
            {
                "var_names": [".+_2$", "custom_name"],
                "filter_vars": "regex",
                "groups": "posterior",
                "var_results": ["parameter_2", "variable_2", "custom_name"],
            },
            {
                "var_names": ["lp"],
                "filter_vars": "regex",
                "groups": "sample_stats",
                "var_results": ["lp"],
            },
        ),
    )
    def test_to_dataframe_selection(self, kwargs):
        results = kwargs.pop("var_results")
        idata = from_dict(
            posterior={
                "parameter_1": np.random.randn(4, 100),
                "parameter_2": np.random.randn(4, 100),
                "variable_1": np.random.randn(4, 100),
                "variable_2": np.random.randn(4, 100),
                "custom_name": np.random.randn(4, 100),
            },
            prior={
                "parameter_1": np.random.randn(4, 100),
                "parameter_2": np.random.randn(4, 100),
            },
            sample_stats={
                "lp": np.random.randn(4, 100),
            },
        )
        test_data = idata.to_dataframe(**kwargs)
        assert not test_data.empty
        assert set(test_data.columns).symmetric_difference(results) == set(["chain", "draw"])

    def test_to_dataframe_bad(self):
        idata = from_dict(
            posterior={"a": np.random.randn(4, 100, 3, 4, 5), "b": np.random.randn(4, 100)},
            sample_stats={"a": np.random.randn(4, 100, 3, 4, 5), "b": np.random.randn(4, 100)},
            observed_data={"a": np.random.randn(3, 4, 5), "b": np.random.randn(4)},
        )
        with pytest.raises(TypeError):
            idata.to_dataframe(index_origin=2)

        with pytest.raises(TypeError):
            idata.to_dataframe(include_coords=False, include_index=False)

        with pytest.raises(TypeError):
            idata.to_dataframe(groups=["observed_data"])

        with pytest.raises(KeyError):
            idata.to_dataframe(groups=["invalid_group"])

        with pytest.raises(ValueError):
            idata.to_dataframe(var_names=["c"])

    @pytest.mark.parametrize("use", (None, "args", "kwargs"))
    def test_map(self, use):
        idata = load_arviz_data("centered_eight")
        args = []
        kwargs = {}
        if use is None:
            fun = lambda x: x + 3
        elif use == "args":
            fun = lambda x, a: x + a
            args = [3]
        else:
            fun = lambda x, a: x + a
            kwargs = {"a": 3}
        groups = ("observed_data", "posterior_predictive")
        idata_map = idata.map(fun, groups, args=args, **kwargs)
        groups_map = idata_map._groups  # pylint: disable=protected-access
        assert groups_map == idata._groups  # pylint: disable=protected-access
        assert np.allclose(
            idata_map.observed_data.obs, fun(idata.observed_data.obs, *args, **kwargs)
        )
        assert np.allclose(
            idata_map.posterior_predictive.obs, fun(idata.posterior_predictive.obs, *args, **kwargs)
        )
        assert np.allclose(idata_map.posterior.mu, idata.posterior.mu)

    def test_repr_html(self):
        """Test if the function _repr_html is generating html."""
        idata = load_arviz_data("centered_eight")
        display_style = OPTIONS["display_style"]
        xr.set_options(display_style="html")
        html = idata._repr_html_()  # pylint: disable=protected-access

        assert html is not None
        assert "<div" in html
        for group in idata._groups:  # pylint: disable=protected-access
            assert group in html
            xr_data = getattr(idata, group)
            for item, _ in xr_data.items():
                assert item in html
        specific_style = ".xr-wrap{width:700px!important;}"
        assert specific_style in html

        xr.set_options(display_style="text")
        html = idata._repr_html_()  # pylint: disable=protected-access
        assert escape(repr(idata)) in html
        xr.set_options(display_style=display_style)

    def test_setitem(self, data_random):
        data_random["new_group"] = data_random.posterior
        assert "new_group" in data_random.groups()
        assert hasattr(data_random, "new_group")

    def test_add_groups(self, data_random):
        data = np.random.normal(size=(4, 500, 8))
        idata = data_random
        idata.add_groups({"prior": {"a": data[..., 0], "b": data}})
        assert "prior" in idata._groups  # pylint: disable=protected-access
        assert isinstance(idata.prior, xr.Dataset)
        assert hasattr(idata, "prior")

        idata.add_groups(warmup_posterior={"a": data[..., 0], "b": data})
        assert "warmup_posterior" in idata._groups_all  # pylint: disable=protected-access
        assert isinstance(idata.warmup_posterior, xr.Dataset)
        assert hasattr(idata, "warmup_posterior")

    def test_add_groups_warning(self, data_random):
        data = np.random.normal(size=(4, 500, 8))
        idata = data_random
        with pytest.warns(UserWarning, match="The group.+not defined in the InferenceData scheme"):
            idata.add_groups({"new_group": idata.posterior}, warn_on_custom_groups=True)
        with pytest.warns(UserWarning, match="the default dims.+will be added automatically"):
            idata.add_groups(constant_data={"a": data[..., 0], "b": data})
        assert idata.new_group.equals(idata.posterior)

    def test_add_groups_error(self, data_random):
        idata = data_random
        with pytest.raises(ValueError, match="One of.+must be provided."):
            idata.add_groups()
        with pytest.raises(ValueError, match="Arguments.+xr.Dataset, xr.Dataarray or dicts"):
            idata.add_groups({"new_group": "new_group"})
        with pytest.raises(ValueError, match="group.+already exists"):
            idata.add_groups({"posterior": idata.posterior})

    def test_extend(self, data_random):
        idata = data_random
        idata2 = create_data_random(
            groups=["prior", "prior_predictive", "observed_data", "warmup_posterior"], seed=7
        )
        idata.extend(idata2)
        assert "prior" in idata._groups_all  # pylint: disable=protected-access
        assert "warmup_posterior" in idata._groups_all  # pylint: disable=protected-access
        assert hasattr(idata, "prior")
        assert hasattr(idata, "prior_predictive")
        assert idata.prior.equals(idata2.prior)
        assert not idata.observed_data.equals(idata2.observed_data)
        assert idata.prior_predictive.equals(idata2.prior_predictive)

        idata.extend(idata2, join="right")
        assert idata.prior.equals(idata2.prior)
        assert idata.observed_data.equals(idata2.observed_data)
        assert idata.prior_predictive.equals(idata2.prior_predictive)

    def test_extend_errors_warnings(self, data_random):
        idata = data_random
        idata2 = create_data_random(groups=["prior", "prior_predictive", "observed_data"], seed=7)
        with pytest.raises(ValueError, match="Extending.+InferenceData objects only."):
            idata.extend("something")
        with pytest.raises(ValueError, match="join must be either"):
            idata.extend(idata2, join="outer")
        idata2.add_groups(new_group=idata2.prior)
        with pytest.warns(UserWarning, match="new_group"):
            idata.extend(idata2, warn_on_custom_groups=True)


class TestNumpyToDataArray:
    def test_1d_dataset(self):
        size = 100
        dataset = convert_to_dataset(np.random.randn(size))
        assert len(dataset.data_vars) == 1

        assert set(dataset.coords) == {"chain", "draw"}
        assert dataset.chain.shape == (1,)
        assert dataset.draw.shape == (size,)

    def test_warns_bad_shape(self):
        # Shape should be (chain, draw, *shape)
        with pytest.warns(UserWarning):
            convert_to_dataset(np.random.randn(100, 4))

    def test_nd_to_dataset(self):
        shape = (1, 2, 3, 4, 5)
        dataset = convert_to_dataset(np.random.randn(*shape))
        assert len(dataset.data_vars) == 1
        var_name = list(dataset.data_vars)[0]

        assert len(dataset.coords) == len(shape)
        assert dataset.chain.shape == shape[:1]
        assert dataset.draw.shape == shape[1:2]
        assert dataset[var_name].shape == shape

    def test_nd_to_inference_data(self):
        shape = (1, 2, 3, 4, 5)
        inference_data = convert_to_inference_data(np.random.randn(*shape), group="prior")
        assert hasattr(inference_data, "prior")
        assert len(inference_data.prior.data_vars) == 1
        var_name = list(inference_data.prior.data_vars)[0]

        assert len(inference_data.prior.coords) == len(shape)
        assert inference_data.prior.chain.shape == shape[:1]
        assert inference_data.prior.draw.shape == shape[1:2]
        assert inference_data.prior[var_name].shape == shape
        assert repr(inference_data).startswith("Inference data with groups")

    def test_more_chains_than_draws(self):
        shape = (10, 4)
        with pytest.warns(UserWarning):
            inference_data = convert_to_inference_data(np.random.randn(*shape), group="prior")
        assert hasattr(inference_data, "prior")
        assert len(inference_data.prior.data_vars) == 1
        var_name = list(inference_data.prior.data_vars)[0]

        assert len(inference_data.prior.coords) == len(shape)
        assert inference_data.prior.chain.shape == shape[:1]
        assert inference_data.prior.draw.shape == shape[1:2]
        assert inference_data.prior[var_name].shape == shape


class TestConvertToDataset:
    @pytest.fixture(scope="class")
    def data(self):
        # pylint: disable=attribute-defined-outside-init
        class Data:
            datadict = {
                "a": np.random.randn(100),
                "b": np.random.randn(1, 100, 10),
                "c": np.random.randn(1, 100, 3, 4),
            }
            coords = {"c1": np.arange(3), "c2": np.arange(4), "b1": np.arange(10)}
            dims = {"b": ["b1"], "c": ["c1", "c2"]}

        return Data

    def test_use_all(self, data):
        dataset = convert_to_dataset(data.datadict, coords=data.coords, dims=data.dims)
        assert set(dataset.data_vars) == {"a", "b", "c"}
        assert set(dataset.coords) == {"chain", "draw", "c1", "c2", "b1"}

        assert set(dataset.a.coords) == {"chain", "draw"}
        assert set(dataset.b.coords) == {"chain", "draw", "b1"}
        assert set(dataset.c.coords) == {"chain", "draw", "c1", "c2"}

    def test_missing_coords(self, data):
        dataset = convert_to_dataset(data.datadict, coords=None, dims=data.dims)
        assert set(dataset.data_vars) == {"a", "b", "c"}
        assert set(dataset.coords) == {"chain", "draw", "c1", "c2", "b1"}

        assert set(dataset.a.coords) == {"chain", "draw"}
        assert set(dataset.b.coords) == {"chain", "draw", "b1"}
        assert set(dataset.c.coords) == {"chain", "draw", "c1", "c2"}

    def test_missing_dims(self, data):
        # missing dims
        coords = {"c_dim_0": np.arange(3), "c_dim_1": np.arange(4), "b_dim_0": np.arange(10)}
        dataset = convert_to_dataset(data.datadict, coords=coords, dims=None)
        assert set(dataset.data_vars) == {"a", "b", "c"}
        assert set(dataset.coords) == {"chain", "draw", "c_dim_0", "c_dim_1", "b_dim_0"}

        assert set(dataset.a.coords) == {"chain", "draw"}
        assert set(dataset.b.coords) == {"chain", "draw", "b_dim_0"}
        assert set(dataset.c.coords) == {"chain", "draw", "c_dim_0", "c_dim_1"}

    def test_skip_dim_0(self, data):
        dims = {"c": [None, "c2"]}
        coords = {"c_dim_0": np.arange(3), "c2": np.arange(4), "b_dim_0": np.arange(10)}
        dataset = convert_to_dataset(data.datadict, coords=coords, dims=dims)
        assert set(dataset.data_vars) == {"a", "b", "c"}
        assert set(dataset.coords) == {"chain", "draw", "c_dim_0", "c2", "b_dim_0"}

        assert set(dataset.a.coords) == {"chain", "draw"}
        assert set(dataset.b.coords) == {"chain", "draw", "b_dim_0"}
        assert set(dataset.c.coords) == {"chain", "draw", "c_dim_0", "c2"}


def test_dict_to_dataset():
    datadict = {"a": np.random.randn(100), "b": np.random.randn(1, 100, 10)}
    dataset = convert_to_dataset(datadict, coords={"c": np.arange(10)}, dims={"b": ["c"]})
    assert set(dataset.data_vars) == {"a", "b"}
    assert set(dataset.coords) == {"chain", "draw", "c"}

    assert set(dataset.a.coords) == {"chain", "draw"}
    assert set(dataset.b.coords) == {"chain", "draw", "c"}


@pytest.mark.skipif(skip_tests, reason="test requires dm-tree which is not installed")
def test_nested_dict_to_dataset():
    datadict = {
        "top": {"a": np.random.randn(100), "b": np.random.randn(1, 100, 10)},
        "d": np.random.randn(100),
    }
    dataset = convert_to_dataset(datadict, coords={"c": np.arange(10)}, dims={("top", "b"): ["c"]})
    assert set(dataset.data_vars) == {("top", "a"), ("top", "b"), "d"}
    assert set(dataset.coords) == {"chain", "draw", "c"}

    assert set(dataset[("top", "a")].coords) == {"chain", "draw"}
    assert set(dataset[("top", "b")].coords) == {"chain", "draw", "c"}
    assert set(dataset.d.coords) == {"chain", "draw"}


def test_dict_to_dataset_event_dims_error():
    datadict = {"a": np.random.randn(1, 100, 10)}
    coords = {"b": np.arange(10), "c": ["x", "y", "z"]}
    msg = "different number of dimensions on data and dims"
    with pytest.raises(ValueError, match=msg):
        convert_to_dataset(datadict, coords=coords, dims={"a": ["b", "c"]})


def test_dict_to_dataset_with_tuple_coord():
    datadict = {"a": np.random.randn(100), "b": np.random.randn(1, 100, 10)}
    dataset = convert_to_dataset(datadict, coords={"c": tuple(range(10))}, dims={"b": ["c"]})
    assert set(dataset.data_vars) == {"a", "b"}
    assert set(dataset.coords) == {"chain", "draw", "c"}

    assert set(dataset.a.coords) == {"chain", "draw"}
    assert set(dataset.b.coords) == {"chain", "draw", "c"}


def test_convert_to_dataset_idempotent():
    first = convert_to_dataset(np.random.randn(100))
    second = convert_to_dataset(first)
    assert first.equals(second)


def test_convert_to_inference_data_idempotent():
    first = convert_to_inference_data(np.random.randn(100), group="prior")
    second = convert_to_inference_data(first)
    assert first.prior is second.prior


def test_convert_to_inference_data_from_file(tmpdir):
    first = convert_to_inference_data(np.random.randn(100), group="prior")
    filename = str(tmpdir.join("test_file.nc"))
    first.to_netcdf(filename)
    second = convert_to_inference_data(filename)
    assert first.prior.equals(second.prior)


def test_convert_to_inference_data_bad():
    with pytest.raises(ValueError):
        convert_to_inference_data(1)


def test_convert_to_dataset_bad(tmpdir):
    first = convert_to_inference_data(np.random.randn(100), group="prior")
    filename = str(tmpdir.join("test_file.nc"))
    first.to_netcdf(filename)
    with pytest.raises(ValueError):
        convert_to_dataset(filename, group="bar")


def test_bad_inference_data():
    with pytest.raises(ValueError):
        InferenceData(posterior=[1, 2, 3])


@pytest.mark.parametrize("warn", [True, False])
def test_inference_data_other_groups(warn):
    datadict = {"a": np.random.randn(100), "b": np.random.randn(1, 100, 10)}
    dataset = convert_to_dataset(datadict, coords={"c": np.arange(10)}, dims={"b": ["c"]})
    if warn:
        with pytest.warns(UserWarning, match="not.+in.+InferenceData scheme"):
            idata = InferenceData(other_group=dataset, warn_on_custom_groups=True)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            idata = InferenceData(other_group=dataset, warn_on_custom_groups=False)
    fails = check_multiple_attrs({"other_group": ["a", "b"]}, idata)
    assert not fails


class TestDataConvert:
    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        class Data:
            # fake 8-school output
            obj = {}
            for key, shape in {"mu": [], "tau": [], "eta": [8], "theta": [8]}.items():
                obj[key] = np.random.randn(chains, draws, *shape)

        return Data

    def get_inference_data(self, data):
        return convert_to_inference_data(
            data.obj,
            group="posterior",
            coords={"school": np.arange(8)},
            dims={"theta": ["school"], "eta": ["school"]},
        )

    def check_var_names_coords_dims(self, dataset):
        assert set(dataset.data_vars) == {"mu", "tau", "eta", "theta"}
        assert set(dataset.coords) == {"chain", "draw", "school"}

    def test_convert_to_inference_data(self, data):
        inference_data = self.get_inference_data(data)
        assert hasattr(inference_data, "posterior")
        self.check_var_names_coords_dims(inference_data.posterior)

    def test_convert_to_dataset(self, draws, chains, data):
        dataset = convert_to_dataset(
            data.obj,
            group="posterior",
            coords={"school": np.arange(8)},
            dims={"theta": ["school"], "eta": ["school"]},
        )
        assert dataset.draw.shape == (draws,)
        assert dataset.chain.shape == (chains,)
        assert dataset.school.shape == (8,)
        assert dataset.theta.shape == (chains, draws, 8)


class TestDataDict:
    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        class Data:
            # fake 8-school output
            obj = {}
            for key, shape in {"mu": [], "tau": [], "eta": [8], "theta": [8]}.items():
                obj[key] = np.random.randn(chains, draws, *shape)

        return Data

    def check_var_names_coords_dims(self, dataset):
        assert set(dataset.data_vars) == {"mu", "tau", "eta", "theta"}
        assert set(dataset.coords) == {"chain", "draw", "school"}

    def get_inference_data(self, data, eight_schools_params, save_warmup=False):
        return from_dict(
            posterior=data.obj,
            posterior_predictive=data.obj,
            sample_stats=data.obj,
            prior=data.obj,
            prior_predictive=data.obj,
            sample_stats_prior=data.obj,
            warmup_posterior=data.obj,
            warmup_posterior_predictive=data.obj,
            predictions=data.obj,
            observed_data=eight_schools_params,
            coords={
                "school": np.arange(8),
            },
            pred_coords={
                "school_pred": np.arange(8),
            },
            dims={"theta": ["school"], "eta": ["school"]},
            pred_dims={"theta": ["school_pred"], "eta": ["school_pred"]},
            save_warmup=save_warmup,
        )

    def test_inference_data(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        test_dict = {
            "posterior": [],
            "prior": [],
            "sample_stats": [],
            "posterior_predictive": [],
            "prior_predictive": [],
            "sample_stats_prior": [],
            "observed_data": ["J", "y", "sigma"],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        self.check_var_names_coords_dims(inference_data.posterior)
        self.check_var_names_coords_dims(inference_data.posterior_predictive)
        self.check_var_names_coords_dims(inference_data.sample_stats)
        self.check_var_names_coords_dims(inference_data.prior)
        self.check_var_names_coords_dims(inference_data.prior_predictive)
        self.check_var_names_coords_dims(inference_data.sample_stats_prior)

        pred_dims = inference_data.predictions.sizes["school_pred"]
        assert pred_dims == 8

    def test_inference_data_warmup(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params, save_warmup=True)
        test_dict = {
            "posterior": [],
            "prior": [],
            "sample_stats": [],
            "posterior_predictive": [],
            "prior_predictive": [],
            "sample_stats_prior": [],
            "observed_data": ["J", "y", "sigma"],
            "warmup_posterior_predictive": [],
            "warmup_posterior": [],
        }
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        self.check_var_names_coords_dims(inference_data.posterior)
        self.check_var_names_coords_dims(inference_data.posterior_predictive)
        self.check_var_names_coords_dims(inference_data.sample_stats)
        self.check_var_names_coords_dims(inference_data.prior)
        self.check_var_names_coords_dims(inference_data.prior_predictive)
        self.check_var_names_coords_dims(inference_data.sample_stats_prior)
        self.check_var_names_coords_dims(inference_data.warmup_posterior)
        self.check_var_names_coords_dims(inference_data.warmup_posterior_predictive)

    def test_inference_data_edge_cases(self):
        # create data
        log_likelihood = {
            "y": np.random.randn(4, 100),
            "log_likelihood": np.random.randn(4, 100, 8),
        }

        # log_likelihood to posterior
        with pytest.warns(UserWarning, match="log_likelihood.+in posterior"):
            assert from_dict(posterior=log_likelihood) is not None

        # dims == None
        assert from_dict(observed_data=log_likelihood, dims=None) is not None

    def test_inference_data_bad(self):
        # create data
        x = np.random.randn(4, 100)

        # input ndarray
        with pytest.raises(TypeError):
            from_dict(posterior=x)
        with pytest.raises(TypeError):
            from_dict(posterior_predictive=x)
        with pytest.raises(TypeError):
            from_dict(sample_stats=x)
        with pytest.raises(TypeError):
            from_dict(prior=x)
        with pytest.raises(TypeError):
            from_dict(prior_predictive=x)
        with pytest.raises(TypeError):
            from_dict(sample_stats_prior=x)
        with pytest.raises(TypeError):
            from_dict(observed_data=x)

    def test_from_dict_warning(self):
        bad_posterior_dict = {"log_likelihood": np.ones((5, 1000, 2))}
        with pytest.warns(UserWarning):
            from_dict(posterior=bad_posterior_dict)


class TestDataNetCDF:
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
            coords={"school": np.array(["a" * i for i in range(8)], dtype="U")},
            dims={"theta": ["school"], "eta": ["school"]},
        )

    def test_io_function(self, data, eight_schools_params):
        # create inference data and assert all attributes are present
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

        # check filename does not exist and save InferenceData
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "..", "saved_models")
        filepath = os.path.join(data_directory, "io_function_testfile.nc")
        # az -function
        to_netcdf(inference_data, filepath)

        # Assert InferenceData has been saved correctly
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
        inference_data2 = from_netcdf(filepath)
        fails = check_multiple_attrs(test_dict, inference_data2)
        assert not fails
        os.remove(filepath)
        assert not os.path.exists(filepath)

    @pytest.mark.parametrize("base_group", ["/", "test_group", "group/subgroup"])
    @pytest.mark.parametrize("groups_arg", [False, True])
    @pytest.mark.parametrize("compress", [True, False])
    @pytest.mark.parametrize("engine", ["h5netcdf", "netcdf4"])
    def test_io_method(self, data, eight_schools_params, groups_arg, base_group, compress, engine):
        # create InferenceData and check it has been properly created
        inference_data = self.get_inference_data(  # pylint: disable=W0612
            data, eight_schools_params
        )
        if engine == "h5netcdf":
            try:
                import h5netcdf  # pylint: disable=unused-import
            except ImportError:
                pytest.skip("h5netcdf not installed")
        elif engine == "netcdf4":
            try:
                import netCDF4  # pylint: disable=unused-import
            except ImportError:
                pytest.skip("netcdf4 not installed")
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

        # check filename does not exist and use to_netcdf method
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "..", "saved_models")
        filepath = os.path.join(data_directory, "io_method_testfile.nc")
        assert not os.path.exists(filepath)
        # InferenceData method
        inference_data.to_netcdf(
            filepath,
            groups=("posterior", "observed_data") if groups_arg else None,
            compress=compress,
            base_group=base_group,
        )

        # assert file has been saved correctly
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
        inference_data2 = InferenceData.from_netcdf(filepath, base_group=base_group)
        if groups_arg:  # if groups arg, update test dict to contain only saved groups
            test_dict = {
                "posterior": ["eta", "theta", "mu", "tau"],
                "observed_data": ["J", "y", "sigma"],
            }
            assert not hasattr(inference_data2, "sample_stats")
        fails = check_multiple_attrs(test_dict, inference_data2)
        assert not fails

        os.remove(filepath)
        assert not os.path.exists(filepath)

    def test_empty_inference_data_object(self):
        inference_data = InferenceData()
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "..", "saved_models")
        filepath = os.path.join(data_directory, "empty_test_file.nc")
        assert not os.path.exists(filepath)
        inference_data.to_netcdf(filepath)
        assert os.path.exists(filepath)
        os.remove(filepath)
        assert not os.path.exists(filepath)


class TestJSON:
    def test_json_converters(self, models):
        idata = models.model_1

        filepath = os.path.realpath("test.json")
        idata.to_json(filepath)

        idata_copy = from_json(filepath)
        for group in idata._groups_all:  # pylint: disable=protected-access
            xr_data = getattr(idata, group)
            test_xr_data = getattr(idata_copy, group)
            assert xr_data.equals(test_xr_data)

        os.remove(filepath)
        assert not os.path.exists(filepath)


class TestDataTree:
    def test_datatree(self):
        idata = load_arviz_data("centered_eight")
        dt = idata.to_datatree()
        idata_back = from_datatree(dt)
        for group, ds in idata.items():
            assert_identical(ds, idata_back[group])
        assert all(group in dt.children for group in idata.groups())

    def test_datatree_attrs(self):
        idata = load_arviz_data("centered_eight")
        idata.attrs = {"not": "empty"}
        assert idata.attrs
        dt = idata.to_datatree()
        idata_back = from_datatree(dt)
        assert dt.attrs == idata.attrs
        assert idata_back.attrs == idata.attrs


class TestConversions:
    def test_id_conversion_idempotent(self):
        stored = load_arviz_data("centered_eight")
        inference_data = convert_to_inference_data(stored)
        assert isinstance(inference_data, InferenceData)
        assert set(inference_data.observed_data.obs.coords["school"].values) == {
            "Hotchkiss",
            "Mt. Hermon",
            "Choate",
            "Deerfield",
            "Phillips Andover",
            "St. Paul's",
            "Lawrenceville",
            "Phillips Exeter",
        }
        assert inference_data.posterior["theta"].dims == ("chain", "draw", "school")

    def test_dataset_conversion_idempotent(self):
        inference_data = load_arviz_data("centered_eight")
        data_set = convert_to_dataset(inference_data.posterior)
        assert isinstance(data_set, xr.Dataset)
        assert set(data_set.coords["school"].values) == {
            "Hotchkiss",
            "Mt. Hermon",
            "Choate",
            "Deerfield",
            "Phillips Andover",
            "St. Paul's",
            "Lawrenceville",
            "Phillips Exeter",
        }
        assert data_set["theta"].dims == ("chain", "draw", "school")

    def test_id_conversion_args(self):
        stored = load_arviz_data("centered_eight")
        IVIES = ["Yale", "Harvard", "MIT", "Princeton", "Cornell", "Dartmouth", "Columbia", "Brown"]
        # test dictionary argument...
        # I reverse engineered a dictionary out of the centered_eight
        # data. That's what this block of code does.
        d = stored.posterior.to_dict()
        d = d["data_vars"]
        test_dict = {}  # type: Dict[str, np.ndarray]
        for var_name in d:
            data = d[var_name]["data"]
            # this is a list of chains that is a list of samples...
            chain_arrs = []
            for chain in data:  # list of samples
                chain_arrs.append(np.array(chain))
            data_arr = np.stack(chain_arrs)
            test_dict[var_name] = data_arr

        inference_data = convert_to_inference_data(
            test_dict, dims={"theta": ["Ivies"]}, coords={"Ivies": IVIES}
        )

        assert isinstance(inference_data, InferenceData)
        assert set(inference_data.posterior.coords["Ivies"].values) == set(IVIES)
        assert inference_data.posterior["theta"].dims == ("chain", "draw", "Ivies")


class TestDataArrayToDataset:
    def test_1d_dataset(self):
        size = 100
        dataset = convert_to_dataset(
            xr.DataArray(np.random.randn(1, size), name="plot", dims=("chain", "draw"))
        )
        assert len(dataset.data_vars) == 1
        assert "plot" in dataset.data_vars
        assert dataset.chain.shape == (1,)
        assert dataset.draw.shape == (size,)

    def test_nd_to_dataset(self):
        shape = (1, 2, 3, 4, 5)
        dataset = convert_to_dataset(
            xr.DataArray(np.random.randn(*shape), dims=("chain", "draw", "dim_0", "dim_1", "dim_2"))
        )
        var_name = list(dataset.data_vars)[0]

        assert len(dataset.data_vars) == 1
        assert dataset.chain.shape == shape[:1]
        assert dataset.draw.shape == shape[1:2]
        assert dataset[var_name].shape == shape

    def test_nd_to_inference_data(self):
        shape = (1, 2, 3, 4, 5)
        inference_data = convert_to_inference_data(
            xr.DataArray(
                np.random.randn(*shape), dims=("chain", "draw", "dim_0", "dim_1", "dim_2")
            ),
            group="prior",
        )
        var_name = list(inference_data.prior.data_vars)[0]

        assert hasattr(inference_data, "prior")
        assert len(inference_data.prior.data_vars) == 1
        assert inference_data.prior.chain.shape == shape[:1]
        assert inference_data.prior.draw.shape == shape[1:2]
        assert inference_data.prior[var_name].shape == shape


class TestExtractDataset:
    def test_default(self):
        idata = load_arviz_data("centered_eight")
        post = extract(idata)
        assert isinstance(post, xr.Dataset)
        assert "sample" in post.dims
        assert post.theta.size == (4 * 500 * 8)

    def test_seed(self):
        idata = load_arviz_data("centered_eight")
        post = extract(idata, rng=7)
        post_pred = extract(idata, group="posterior_predictive", rng=7)
        assert all(post.sample == post_pred.sample)

    def test_no_combine(self):
        idata = load_arviz_data("centered_eight")
        post = extract(idata, combined=False)
        assert "sample" not in post.dims
        assert post.sizes["chain"] == 4
        assert post.sizes["draw"] == 500

    def test_var_name_group(self):
        idata = load_arviz_data("centered_eight")
        prior = extract(idata, group="prior", var_names="the", filter_vars="like")
        assert {} == prior.attrs
        assert "theta" in prior.name

    def test_keep_dataset(self):
        idata = load_arviz_data("centered_eight")
        prior = extract(
            idata, group="prior", var_names="the", filter_vars="like", keep_dataset=True
        )
        assert prior.attrs == idata.prior.attrs
        assert "theta" in prior.data_vars
        assert "mu" not in prior.data_vars

    def test_subset_samples(self):
        idata = load_arviz_data("centered_eight")
        post = extract(idata, num_samples=10)
        assert post.sizes["sample"] == 10
        assert post.attrs == idata.posterior.attrs


def test_convert_to_inference_data_with_array_like():
    class ArrayLike:
        def __init__(self, data):
            self._data = np.asarray(data)

        def __array__(self):
            return self._data

    array_like = ArrayLike(np.random.randn(4, 100))
    idata = convert_to_inference_data(array_like, group="posterior")

    assert hasattr(idata, "posterior")
    assert "x" in idata.posterior.data_vars
    assert idata.posterior["x"].shape == (4, 100)
