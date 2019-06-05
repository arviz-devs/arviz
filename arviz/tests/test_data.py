# pylint: disable=no-member, invalid-name, redefined-outer-name
# pylint: disable=too-many-lines
from collections import namedtuple
import os
from urllib.parse import urlunsplit
import numpy as np
import pytest

from arviz import (
    concat,
    convert_to_inference_data,
    convert_to_dataset,
    from_dict,
    from_netcdf,
    to_netcdf,
    load_arviz_data,
    list_datasets,
    clear_data_home,
    InferenceData,
)
from ..data.base import generate_dims_coords, make_attrs
from ..data.datasets import REMOTE_DATASETS, LOCAL_DATASETS, RemoteFileMetadata
from .helpers import (  # pylint: disable=unused-import
    chains,
    check_multiple_attrs,
    draws,
    eight_schools_params,
)


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
            filename=filename,
            url=url,
            checksum="9ae00c83654b3f061d32c882ec0a270d10838fa36515ecb162b89a290e014849",
            description=centered.description,
        ),
    )
    monkeypatch.setitem(
        REMOTE_DATASETS,
        "bad_checksum",
        RemoteFileMetadata(
            filename=filename, url=url, checksum="bad!", description=centered.description
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
    assert load_arviz_data("centered_eight")


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


def test_dims_coords_extra_dims():
    shape = 4, 20
    var_name = "x"
    with pytest.warns(SyntaxWarning):
        dims, coords = generate_dims_coords(shape, var_name, dims=["xx", "xy", "xz"])
    assert "xx" in dims
    assert "xy" in dims
    assert "xz" in dims
    assert len(coords["xx"]) == 4
    assert len(coords["xy"]) == 20


def test_make_attrs():
    extra_attrs = {"key": "Value"}
    attrs = make_attrs(attrs=extra_attrs)
    assert "key" in attrs
    assert attrs["key"] == "Value"


def test_addition():
    idata1 = from_dict(
        posterior={"A": np.random.randn(2, 10, 2), "B": np.random.randn(2, 10, 5, 2)}
    )
    idata2 = from_dict(prior={"C": np.random.randn(2, 10, 2), "D": np.random.randn(2, 10, 5, 2)})
    new_idata = idata1 + idata2
    assert new_idata is not None
    test_dict = {"posterior": ["A", "B"], "prior": ["C", "D"]}
    fails = check_multiple_attrs(test_dict, new_idata)
    assert not fails


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("sequence", [True, False])
def test_concat(copy, inplace, sequence):
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
    with pytest.raises(TypeError):
        concat(idata, np.array([1, 2, 3, 4, 5]))
    with pytest.raises(NotImplementedError):
        concat(idata, idata)


@pytest.mark.parametrize("inplace", [True, False])
def test_sel_method(inplace):
    data = np.random.normal(size=(4, 500, 8))
    idata = from_dict(
        posterior={"a": data[..., 0], "b": data},
        sample_stats={"a": data[..., 0], "b": data},
        observed_data={"b": data[0, 0, :]},
        posterior_predictive={"a": data[..., 0], "b": data},
    )
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
        with pytest.warns(SyntaxWarning):
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
        inference_data = convert_to_inference_data(np.random.randn(*shape), group="foo")
        assert hasattr(inference_data, "foo")
        assert len(inference_data.foo.data_vars) == 1
        var_name = list(inference_data.foo.data_vars)[0]

        assert len(inference_data.foo.coords) == len(shape)
        assert inference_data.foo.chain.shape == shape[:1]
        assert inference_data.foo.draw.shape == shape[1:2]
        assert inference_data.foo[var_name].shape == shape
        assert repr(inference_data).startswith("Inference data with groups")

    def test_more_chains_than_draws(self):
        shape = (10, 4)
        with pytest.warns(SyntaxWarning):
            inference_data = convert_to_inference_data(np.random.randn(*shape), group="foo")
        assert hasattr(inference_data, "foo")
        assert len(inference_data.foo.data_vars) == 1
        var_name = list(inference_data.foo.data_vars)[0]

        assert len(inference_data.foo.coords) == len(shape)
        assert inference_data.foo.chain.shape == shape[:1]
        assert inference_data.foo.draw.shape == shape[1:2]
        assert inference_data.foo[var_name].shape == shape


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


def test_convert_to_dataset_idempotent():
    first = convert_to_dataset(np.random.randn(100))
    second = convert_to_dataset(first)
    assert first.equals(second)


def test_convert_to_inference_data_idempotent():
    first = convert_to_inference_data(np.random.randn(100), group="foo")
    second = convert_to_inference_data(first)
    assert first.foo is second.foo


def test_convert_to_inference_data_from_file(tmpdir):
    first = convert_to_inference_data(np.random.randn(100), group="foo")
    filename = str(tmpdir.join("test_file.nc"))
    first.to_netcdf(filename)
    second = convert_to_inference_data(filename)
    assert first.foo.equals(second.foo)


def test_convert_to_inference_data_bad():
    with pytest.raises(ValueError):
        convert_to_inference_data(1)


def test_convert_to_dataset_bad(tmpdir):
    first = convert_to_inference_data(np.random.randn(100), group="foo")
    filename = str(tmpdir.join("test_file.nc"))
    first.to_netcdf(filename)
    with pytest.raises(ValueError):
        convert_to_dataset(filename, group="bar")


def test_bad_inference_data():
    with pytest.raises(ValueError):
        InferenceData(posterior=[1, 2, 3])


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

    def test_inference_data_edge_cases(self):
        # create data
        log_likelihood = {
            "y": np.random.randn(4, 100),
            "log_likelihood": np.random.randn(4, 100, 8),
        }

        # log_likelihood to posterior
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
        with pytest.warns(SyntaxWarning):
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
            coords={"school": np.arange(8)},
            dims={"theta": ["school"], "eta": ["school"]},
        )

    def test_io_function(self, data, eight_schools_params):
        inference_data = self.get_inference_data(  # pylint: disable=W0612
            data, eight_schools_params
        )
        assert hasattr(inference_data, "posterior")
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "saved_models")
        filepath = os.path.join(data_directory, "io_function_testfile.nc")
        # az -function
        to_netcdf(inference_data, filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
        inference_data2 = from_netcdf(filepath)
        assert hasattr(inference_data2, "posterior")
        os.remove(filepath)
        assert not os.path.exists(filepath)

    def test_io_method(self, data, eight_schools_params):
        inference_data = self.get_inference_data(  # pylint: disable=W0612
            data, eight_schools_params
        )
        assert hasattr(inference_data, "posterior")
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "saved_models")
        filepath = os.path.join(data_directory, "io_method_testfile.nc")
        assert not os.path.exists(filepath)
        # InferenceData method
        inference_data.to_netcdf(filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
        inference_data2 = InferenceData.from_netcdf(filepath)
        assert hasattr(inference_data2, "posterior")
        os.remove(filepath)
        assert not os.path.exists(filepath)

    def test_empty_inference_data_object(self):
        inference_data = InferenceData()
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "saved_models")
        filepath = os.path.join(data_directory, "empty_test_file.nc")
        assert not os.path.exists(filepath)
        inference_data.to_netcdf(filepath)
        assert os.path.exists(filepath)
        os.remove(filepath)
        assert not os.path.exists(filepath)
