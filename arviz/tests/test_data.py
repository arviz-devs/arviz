# pylint: disable=no-member, invalid-name, redefined-outer-name
# pylint: disable=too-many-lines
from collections import namedtuple
import os
from urllib.parse import urlunsplit

import numpy as np
import pymc3 as pm
import pytest

from arviz import (
    convert_to_inference_data,
    convert_to_dataset,
    from_cmdstan,
    from_dict,
    from_pymc3,
    from_pystan,
    from_pyro,
    from_emcee,
    from_netcdf,
    to_netcdf,
    load_data,
    save_data,
    load_arviz_data,
    list_datasets,
    clear_data_home,
    InferenceData,
)
from ..data.base import generate_dims_coords, make_attrs
from ..data.io_pystan import get_draws, get_draws_stan3  # pylint: disable=unused-import
from ..data.datasets import REMOTE_DATASETS, LOCAL_DATASETS, RemoteFileMetadata
from .helpers import (  # pylint: disable=unused-import
    _emcee_lnprior as emcee_lnprior,
    _emcee_lnprob as emcee_lnprob,
    eight_schools_params,
    load_cached_models,
    pystan_extract_unpermuted,
    stan_extract_dict,
    pystan_version,
)


@pytest.fixture(scope="module")
def draws():
    return 500


@pytest.fixture(scope="module")
def chains():
    return 2


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


class TestDictNetCDFUtils:
    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        # Data of the Eight Schools Model

        class Data:
            _, stan_fit = load_cached_models(eight_schools_params, draws, chains)["pystan"]
            if pystan_version() == 2:
                stan_dict = pystan_extract_unpermuted(stan_fit)
                obj = {}
                for name, vals in stan_dict.items():
                    if name not in {"y_hat", "log_lik"}:  # extra vars
                        obj[name] = np.swapaxes(vals, 0, 1)
            else:
                stan_dict = stan_extract_dict(stan_fit)
                obj = {}
                for name, vals in stan_dict.items():
                    if name not in {"y_hat", "log_lik"}:  # extra vars
                        obj[name] = vals

        return Data

    def check_var_names_coords_dims(self, dataset):
        assert set(dataset.data_vars) == {"mu", "tau", "eta", "theta"}
        assert set(dataset.coords) == {"chain", "draw", "school"}

    def get_inference_data(self, data, eight_schools_params):
        return convert_to_inference_data(
            data.obj,
            group="posterior",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
        )

    def test_testing_extract(self, data):
        if pystan_version() == 2:
            extract_func = pystan_extract_unpermuted
            parameters = data.stan_fit.model_pars
        else:
            extract_func = stan_extract_dict
            parameters = data.stan_fit.param_names
        assert isinstance(extract_func(data.stan_fit, var_names=None), dict)
        assert isinstance(extract_func(data.stan_fit, var_names=parameters[0]), dict)
        assert isinstance(extract_func(data.stan_fit, var_names=parameters), dict)
        assert isinstance(
            extract_func(data.stan_fit, var_names=[parameters[0], parameters[0]]), dict
        )

    def test_convert_to_inference_data(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        assert hasattr(inference_data, "posterior")
        self.check_var_names_coords_dims(inference_data.posterior)

    def test_convert_to_dataset(self, eight_schools_params, draws, chains, data):
        dataset = convert_to_dataset(
            data.obj,
            group="posterior",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
        )
        assert dataset.draw.shape == (draws,)
        assert dataset.chain.shape == (chains,)
        assert dataset.school.shape == (eight_schools_params["J"],)
        assert dataset.theta.shape == (chains, draws, eight_schools_params["J"])


class TestDictIONetCDFUtils:
    @pytest.fixture(scope="class")
    def data(self, eight_schools_params, draws, chains):
        # Data of the Eight Schools Model

        class Data:
            _, stan_fit = load_cached_models(eight_schools_params, draws, chains)["pystan"]
            if pystan_version() == 2:
                stan_dict = pystan_extract_unpermuted(stan_fit)
                obj = {}
                for name, vals in stan_dict.items():
                    if name not in {"y_hat", "log_lik"}:  # extra vars
                        obj[name] = np.swapaxes(vals, 0, 1)
            else:
                stan_dict = stan_extract_dict(stan_fit)
                obj = {}
                for name, vals in stan_dict.items():
                    if name not in {"y_hat", "log_lik"}:  # extra vars
                        obj[name] = vals

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
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
        )

    def test_inference_data(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        assert hasattr(inference_data, "posterior")
        assert hasattr(inference_data, "posterior_predictive")
        assert hasattr(inference_data, "sample_stats")
        assert hasattr(inference_data, "prior")
        assert hasattr(inference_data, "prior_predictive")
        assert hasattr(inference_data, "sample_stats_prior")
        assert hasattr(inference_data, "observed_data")
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


class TestEmceeNetCDFUtils:
    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        class Data:
            # chains are not used
            # emcee uses lots of walkers
            obj = load_cached_models(eight_schools_params, draws, chains)["emcee"]

        return Data

    def get_inference_data(self, data):
        return from_emcee(data.obj, var_names=["ln(f)", "b", "m"])

    def test_inference_data(self, data):
        inference_data = self.get_inference_data(data)
        assert hasattr(inference_data, "posterior")
        assert hasattr(inference_data.posterior, "ln(f)")
        assert hasattr(inference_data.posterior, "b")
        assert hasattr(inference_data.posterior, "m")

    def test_verify_var_names(self, data):
        with pytest.raises(ValueError):
            from_emcee(data.obj, var_names=["not", "enough"])

    def test_verify_arg_names(self, data):
        with pytest.raises(ValueError):
            from_emcee(data.obj, arg_names=["not", "enough"])

    def test_ln_funcs_for_infinity(self):
        assert np.isinf(emcee_lnprior([1000, 10000, 1_000_000]))
        assert np.isinf(emcee_lnprob([1000, 10000, 1_000_000], 0, 0, 0))


class TestIONetCDFUtils:
    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        class Data:
            model, obj = load_cached_models(eight_schools_params, draws, chains)["pymc3"]

        return Data

    def get_inference_data(self, data, eight_schools_params):  # pylint: disable=W0613
        with data.model:
            prior = pm.sample_prior_predictive()
            posterior_predictive = pm.sample_posterior_predictive(data.obj)

        return from_pymc3(
            trace=data.obj,
            prior=prior,
            posterior_predictive=posterior_predictive,
            coords={"school": np.arange(eight_schools_params["J"])},
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
        # Test deprecated functions
        save_data(inference_data, filepath)
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
        inference_data3 = load_data(filepath)
        assert hasattr(inference_data3, "posterior")
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


class TestPyMC3NetCDFUtils:
    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        class Data:
            model, obj = load_cached_models(eight_schools_params, draws, chains)["pymc3"]

        return Data

    def get_inference_data(self, data, eight_schools_params):
        with data.model:
            prior = pm.sample_prior_predictive()
            posterior_predictive = pm.sample_posterior_predictive(data.obj)

        return from_pymc3(
            trace=data.obj,
            prior=prior,
            posterior_predictive=posterior_predictive,
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "eta": ["school"]},
        )

    def test_posterior(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        assert hasattr(inference_data, "posterior")

    def test_sampler_stats(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        assert hasattr(inference_data, "sample_stats")

    def test_posterior_predictive(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        assert hasattr(inference_data, "posterior_predictive")

    def test_prior(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        assert hasattr(inference_data, "prior")


class TestPyroNetCDFUtils:
    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        class Data:
            obj = load_cached_models(eight_schools_params, draws, chains)["pyro"]

        return Data

    def get_inference_data(self, data):
        return from_pyro(posterior=data.obj)

    def test_inference_data(self, data):
        inference_data = self.get_inference_data(data)
        assert hasattr(inference_data, "posterior")


class TestPyStanNetCDFUtils:
    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        class Data:
            model, obj = load_cached_models(eight_schools_params, draws, chains)["pystan"]

        return Data

    def get_inference_data(self, data, eight_schools_params):
        """vars as str."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive="y_hat",
            prior=data.obj,
            prior_predictive="y_hat",
            observed_data="y",
            log_likelihood="log_lik",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
            },
            posterior_model=data.model,
            prior_model=data.model,
        )

    def get_inference_data2(self, data, eight_schools_params):
        """vars as lists."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive=["y_hat"],
            prior=data.obj,
            prior_predictive=["y_hat"],
            observed_data="y",
            log_likelihood="log_lik",
            coords={
                "school": np.arange(eight_schools_params["J"]),
                "log_likelihood_dim": np.arange(eight_schools_params["J"]),
            },
            dims={
                "theta": ["school"],
                "y": ["school"],
                "y_hat": ["school"],
                "eta": ["school"],
                "log_lik": ["log_likelihood_dim"],
            },
            posterior_model=data.model,
            prior_model=data.model,
        )

    def get_inference_data3(self, data, eight_schools_params):
        """multiple vars as lists."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive=["y_hat", "log_lik"],
            prior=data.obj,
            prior_predictive=["y_hat", "log_lik"],
            observed_data="y",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "y": ["school"], "y_hat": ["school"], "eta": ["school"]},
            posterior_model=data.model,
            prior_model=data.model,
        )

    def get_inference_data4(self, data):
        """multiple vars as lists."""
        return from_pystan(
            posterior=data.obj,
            posterior_predictive=None,
            prior=data.obj,
            prior_predictive=None,
            observed_data="y",
            coords=None,
            dims=None,
            posterior_model=data.model,
            prior_model=data.model,
        )

    def test_sampler_stats(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        assert hasattr(inference_data, "sample_stats")
        assert hasattr(inference_data.sample_stats, "lp")
        assert hasattr(inference_data.sample_stats, "diverging")

    def test_inference_data(self, data, eight_schools_params):
        inference_data1 = self.get_inference_data(data, eight_schools_params)
        inference_data2 = self.get_inference_data2(data, eight_schools_params)
        inference_data3 = self.get_inference_data3(data, eight_schools_params)
        inference_data4 = self.get_inference_data4(data)
        # inference_data 1
        assert hasattr(inference_data1.sample_stats, "log_likelihood")
        assert hasattr(inference_data1.posterior, "theta")
        assert hasattr(inference_data1.prior, "theta")
        assert hasattr(inference_data1.observed_data, "y")
        # inference_data 2
        assert hasattr(inference_data2.posterior_predictive, "y_hat")
        assert hasattr(inference_data2.prior_predictive, "y_hat")
        assert hasattr(inference_data2.sample_stats, "lp")
        assert hasattr(inference_data2.sample_stats_prior, "lp")
        assert hasattr(inference_data2.observed_data, "y")
        # inference_data 3
        assert hasattr(inference_data3.posterior_predictive, "y_hat")
        assert hasattr(inference_data3.prior_predictive, "y_hat")
        assert hasattr(inference_data3.sample_stats, "lp")
        assert hasattr(inference_data3.sample_stats_prior, "lp")
        assert hasattr(inference_data3.observed_data, "y")
        # inference_data 4
        assert hasattr(inference_data4.posterior, "theta")
        assert hasattr(inference_data4.prior, "theta")

    def test_invalid_fit(self, data):
        if pystan_version() == 2:
            model = data.model
            model_data = {
                "J": 8,
                "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
                "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
            }
            fit_test_grad = model.sampling(
                data=model_data, test_grad=True, check_hmc_diagnostics=False
            )
            with pytest.raises(AttributeError):
                _ = from_pystan(posterior=fit_test_grad)
            fit = model.sampling(data=model_data, iter=100, chains=1, check_hmc_diagnostics=False)
            del fit.sim["samples"]
            with pytest.raises(AttributeError):
                _ = from_pystan(posterior=fit)

    def test_empty_parameter(self):
        if pystan_version() == 2:
            model_code = """
                parameters {
                    real y;
                    vector[0] z;
                }
                model {
                    y ~ normal(0,1);
                }
            """
            from pystan import StanModel

            model = StanModel(model_code=model_code)
            fit = model.sampling(iter=10, chains=2, check_hmc_diagnostics=False)
            posterior = from_pystan(posterior=fit)
            assert hasattr(posterior, "posterior")
            assert hasattr(posterior.posterior, "y")
            assert not hasattr(posterior.posterior, "z")

    def test_get_draws(self, data):
        fit = data.obj
        if pystan_version() == 2:
            draws = get_draws(fit, variables=["theta", "theta"])
            assert draws.get("theta") is not None
        else:
            draws = get_draws_stan3(fit, variables=["theta", "theta"])
            assert draws.get("theta") is not None


class TestTfpNetCDFUtils:
    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        class Data:
            # Returns result of from_tfp
            obj = load_cached_models({}, draws, chains)[  # pylint: disable=no-value-for-parameter
                "tensorflow_probability"
            ]

        return Data

    def get_inference_data(self, data):
        return data.obj

    def test_inference_data(self, data):
        inference_data = self.get_inference_data(data)
        assert hasattr(inference_data, "posterior")


class TestCmdStanNetCDFUtils:
    @pytest.fixture(scope="session")
    def data_directory(self):
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "saved_models")
        return data_directory

    @pytest.fixture(scope="class")
    def paths(self, data_directory):
        paths = {
            "no_warmup": [
                os.path.join(data_directory, "cmdstan/output_no_warmup1.csv"),
                os.path.join(data_directory, "cmdstan/output_no_warmup2.csv"),
                os.path.join(data_directory, "cmdstan/output_no_warmup3.csv"),
                os.path.join(data_directory, "cmdstan/output_no_warmup4.csv"),
            ],
            "warmup": [
                os.path.join(data_directory, "cmdstan/output_warmup1.csv"),
                os.path.join(data_directory, "cmdstan/output_warmup2.csv"),
                os.path.join(data_directory, "cmdstan/output_warmup3.csv"),
                os.path.join(data_directory, "cmdstan/output_warmup4.csv"),
            ],
            "no_warmup_glob": os.path.join(data_directory, "cmdstan/output_no_warmup[0-9].csv"),
            "warmup_glob": os.path.join(data_directory, "cmdstan/output_warmup[0-9].csv"),
            "combined_no_warmup": [
                os.path.join(data_directory, "cmdstan/combined_output_no_warmup.csv")
            ],
            "combined_warmup": [os.path.join(data_directory, "cmdstan/combined_output_warmup.csv")],
            "combined_no_warmup_glob": os.path.join(
                data_directory, "cmdstan/combined_output_no_warmup.csv"
            ),
            "combined_warmup_glob": os.path.join(
                data_directory, "cmdstan/combined_output_warmup.csv"
            ),
            "eight_schools_glob": os.path.join(
                data_directory, "cmdstan/eight_schools_output[0-9].csv"
            ),
            "eight_schools": [
                os.path.join(data_directory, "cmdstan/eight_schools_output1.csv"),
                os.path.join(data_directory, "cmdstan/eight_schools_output2.csv"),
                os.path.join(data_directory, "cmdstan/eight_schools_output3.csv"),
                os.path.join(data_directory, "cmdstan/eight_schools_output4.csv"),
            ],
            "missing_files": [
                os.path.join(data_directory, "cmdstan/combined_missing_config.csv"),
                os.path.join(data_directory, "cmdstan/combined_missing_adaptation.csv"),
                os.path.join(data_directory, "cmdstan/combined_missing_timing1.csv"),
                os.path.join(data_directory, "cmdstan/combined_missing_timing2.csv"),
            ],
        }
        return paths

    @pytest.fixture(scope="class")
    def observed_data_paths(self, data_directory):
        observed_data_paths = [
            os.path.join(data_directory, "cmdstan/eight_schools.data.R"),
            os.path.join(data_directory, "cmdstan/example_stan.data.R"),
        ]

        return observed_data_paths

    def get_inference_data(self, posterior, **kwargs):
        return from_cmdstan(posterior=posterior, **kwargs)

    def test_sample_stats(self, paths):
        for key, path in paths.items():
            if "missing" in key:
                continue
            inference_data = self.get_inference_data(path)
            assert hasattr(inference_data, "sample_stats")

    def test_inference_data_shapes(self, paths):
        """Assert that shapes are transformed correctly"""
        for key, path in paths.items():
            if "eight" in key or "missing" in key:
                continue
            inference_data = self.get_inference_data(path)
            assert hasattr(inference_data, "posterior")
            assert hasattr(inference_data.posterior, "y")
            assert hasattr(inference_data.posterior, "x")
            assert hasattr(inference_data.posterior, "Z")
            assert inference_data.posterior["y"].shape == (4, 100)
            assert inference_data.posterior["x"].shape == (4, 100, 3)
            assert inference_data.posterior["Z"].shape == (4, 100, 4, 6)
            dims = ["chain", "draw"]
            y_mean_true = 0
            y_mean = inference_data.posterior["y"].mean(dim=dims)
            assert np.isclose(y_mean, y_mean_true, atol=1e-1)
            x_mean_true = np.array([1, 2, 3])
            x_mean = inference_data.posterior["x"].mean(dim=dims)
            assert np.isclose(x_mean, x_mean_true, atol=1e-1).all()
            Z_mean_true = np.array([1, 2, 3, 4])
            Z_mean = inference_data.posterior["Z"].mean(dim=dims).mean(axis=1)
            assert np.isclose(Z_mean, Z_mean_true, atol=7e-1).all()

    def test_inference_data_input_types1(self, paths, observed_data_paths):
        """Check input types

            posterior --> str, list of str
            prior --> str, list of str
            posterior_predictive --> str, variable in posterior
            observed_data --> Rdump format
            observed_data_var --> str, variable
            log_likelihood --> str
            coords --> one to many
            dims --> one to many
        """
        for key, path in paths.items():
            if "eight" not in key:
                continue
            inference_data = self.get_inference_data(
                posterior=path,
                posterior_predictive="y_hat",
                prior=path,
                prior_predictive="y_hat",
                observed_data=observed_data_paths[0],
                observed_data_var="y",
                log_likelihood="log_lik",
                coords={"school": np.arange(8)},
                dims={
                    "theta": ["school"],
                    "y": ["school"],
                    "log_lik": ["school"],
                    "y_hat": ["school"],
                    "eta": ["school"],
                },
            )
            assert hasattr(inference_data, "posterior")
            assert hasattr(inference_data, "sample_stats")
            assert hasattr(inference_data.sample_stats, "log_likelihood")
            assert hasattr(inference_data, "posterior_predictive")
            assert hasattr(inference_data, "observed_data")

    def test_inference_data_input_types2(self, paths, observed_data_paths):
        """Check input types (change, see earlier)

            posterior_predictive --> List[str], variable in posterior
            observed_data_var --> List[str], variable
        """
        for key, path in paths.items():
            if "eight" not in key:
                continue
            inference_data = self.get_inference_data(
                posterior=path,
                posterior_predictive=["y_hat"],
                prior=path,
                prior_predictive=["y_hat"],
                observed_data=observed_data_paths[0],
                observed_data_var=["y"],
                log_likelihood="log_lik",
                coords={"school": np.arange(8)},
                dims={
                    "theta": ["school"],
                    "y": ["school"],
                    "log_lik": ["school"],
                    "y_hat": ["school"],
                    "eta": ["school"],
                },
            )
            assert hasattr(inference_data, "posterior")
            assert hasattr(inference_data, "sample_stats")
            assert hasattr(inference_data.sample_stats, "log_likelihood")
            assert hasattr(inference_data, "posterior_predictive")
            assert hasattr(inference_data, "observed_data")

    def test_inference_data_input_types3(self, paths, observed_data_paths):
        """Check input types (change, see earlier)

            posterior_predictive --> str, csv file
            coords --> one to many + one to one (default dim)
            dims --> one to many
        """
        for key, path in paths.items():
            if "eight" not in key:
                continue
            post_pred = paths["eight_schools_glob"]
            inference_data = self.get_inference_data(
                posterior=path,
                posterior_predictive=post_pred,
                prior=path,
                prior_predictive=post_pred,
                observed_data=observed_data_paths[0],
                observed_data_var=["y"],
                log_likelihood="log_lik",
                coords={"school": np.arange(8), "log_lik_dim_0": np.arange(8)},
                dims={"theta": ["school"], "y": ["school"], "y_hat": ["school"], "eta": ["school"]},
            )
            assert hasattr(inference_data, "posterior")
            assert hasattr(inference_data, "sample_stats")
            assert hasattr(inference_data.sample_stats, "log_likelihood")
            assert hasattr(inference_data, "posterior_predictive")
            assert hasattr(inference_data, "observed_data")

    def test_inference_data_input_types4(self, paths):
        """Check input types (change, see earlier)

            coords --> one to many + one to one (non-default dim)
            dims --> one to many + one to one
        """

        path = paths["combined_no_warmup"]
        for path in [path, path[0]]:
            inference_data = self.get_inference_data(
                posterior=path,
                posterior_predictive=path,
                prior=path,
                prior_predictive=path,
                observed_data=None,
                observed_data_var=None,
                coords={"rand": np.arange(3)},
                dims={"x": ["rand"]},
            )
            assert hasattr(inference_data, "posterior")
            assert hasattr(inference_data, "sample_stats")
            assert hasattr(inference_data, "posterior_predictive")
            assert hasattr(inference_data, "prior")
            assert hasattr(inference_data, "sample_stats_prior")
            assert hasattr(inference_data, "prior_predictive")

    def test_inference_data_input_types5(self, paths, observed_data_paths):
        """Check input types (change, see earlier)

            posterior_predictive is None
            prior_predictive is None
        """
        for key, path in paths.items():
            if "eight" not in key:
                continue
            inference_data = self.get_inference_data(
                posterior=path,
                posterior_predictive=None,
                prior=path,
                prior_predictive=None,
                observed_data=observed_data_paths[0],
                observed_data_var=["y"],
                log_likelihood=["log_lik"],
                coords={"school": np.arange(8), "log_lik_dim": np.arange(8)},
                dims={
                    "theta": ["school"],
                    "y": ["school"],
                    "log_lik": ["log_lik_dim"],
                    "y_hat": ["school"],
                    "eta": ["school"],
                },
            )
            assert hasattr(inference_data, "posterior")
            assert hasattr(inference_data, "sample_stats")
            assert hasattr(inference_data.sample_stats, "log_likelihood")
            assert hasattr(inference_data, "observed_data")

    def test_inference_data_bad_csv(self, paths):
        """Check ValueError for csv with missing headers"""
        for key, _paths in paths.items():
            if "missing" not in key:
                continue
            for path in _paths:
                with pytest.raises(ValueError):
                    self.get_inference_data(posterior=path)

    def test_inference_data_observed_data1(self, observed_data_paths):
        """Read Rdump, check shapes are correct

            All variables
        """
        path = observed_data_paths[1]
        inference_data = self.get_inference_data(posterior=None, observed_data=path)
        assert hasattr(inference_data, "observed_data")
        assert len(inference_data.observed_data.data_vars) == 3
        assert inference_data.observed_data["x"].shape == (1,)
        assert inference_data.observed_data["y"].shape == (3,)
        assert inference_data.observed_data["Z"].shape == (4, 5)

    def test_inference_data_observed_data2(self, observed_data_paths):
        """Read Rdump, check shapes are correct

            One variable as str
        """
        path = observed_data_paths[1]
        inference_data = self.get_inference_data(
            posterior=None, observed_data=path, observed_data_var="x"
        )
        assert hasattr(inference_data, "observed_data")
        assert len(inference_data.observed_data.data_vars) == 1
        assert inference_data.observed_data["x"].shape == (1,)

    def test_inference_data_observed_data3(self, observed_data_paths):
        """Read Rdump, check shapes are correct

            One variable as a list
        """
        path = observed_data_paths[1]
        inference_data = self.get_inference_data(
            posterior=None, observed_data=path, observed_data_var=["x"]
        )
        assert hasattr(inference_data, "observed_data")
        assert len(inference_data.observed_data.data_vars) == 1
        assert inference_data.observed_data["x"].shape == (1,)

    def test_inference_data_observed_data4(self, observed_data_paths):
        """Read Rdump, check shapes are correct

            Many variables as list
        """
        path = observed_data_paths[1]
        inference_data = self.get_inference_data(
            posterior=None, observed_data=path, observed_data_var=["y", "Z"]
        )
        assert hasattr(inference_data, "observed_data")
        assert len(inference_data.observed_data.data_vars) == 2
        assert inference_data.observed_data["y"].shape == (3,)
        assert inference_data.observed_data["Z"].shape == (4, 5)
