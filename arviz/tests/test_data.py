# pylint: disable=no-member,invalid-name
import os
import numpy as np
import pymc3 as pm
import pytest

from arviz import (
    convert_to_inference_data,
    convert_to_dataset,
    from_cmdstan,
    from_pymc3,
    from_pystan,
    from_emcee,
)
from .helpers import (
    eight_schools_params,
    load_cached_models,
    BaseArvizTest,
    pystan_extract_unpermuted,
    pystan_extract_normal,
)


@pytest.fixture(scope="class")
def draws():
    return 500


@pytest.fixture(scope="class")
def chains():
    return 2


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


class TestDictNetCDFUtils(BaseArvizTest):

    @pytest.fixture(scope="class")
    def data(self, draws, chains):
        # Data of the Eight Schools Model

        class Data:
            _, stan_fit = load_cached_models(draws, chains)["pystan"]
            stan_dict = pystan_extract_unpermuted(stan_fit)
            obj = {}
            for name, vals in stan_dict.items():
                if name not in {"y_hat", "log_lik"}:  # extra vars
                    obj[name] = np.swapaxes(vals, 0, 1)
        return Data

    def check_var_names_coords_dims(self, dataset):
        assert set(dataset.data_vars) == {"mu", "tau", "theta_tilde", "theta"}
        assert set(dataset.coords) == {"chain", "draw", "school"}

    def get_inference_data(self, data, eight_schools_params):
        return convert_to_inference_data(
            data.obj,
            group="posterior",
            coords={"school": np.arange(eight_schools_params["J"])},
            dims={"theta": ["school"], "theta_tilde": ["school"]},
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
            dims={"theta": ["school"], "theta_tilde": ["school"]},
        )
        assert dataset.draw.shape == (draws,)
        assert dataset.chain.shape == (chains,)
        assert dataset.school.shape == (eight_schools_params["J"],)
        assert dataset.theta.shape == (chains, draws, eight_schools_params["J"])


class TestEmceeNetCDFUtils:

    @pytest.fixture(scope="class")
    def obj(self, draws):
        fake_chains = 2  # emcee uses lots of walkers
        obj = load_cached_models(draws, fake_chains)["emcee"]
        return obj

    def get_inference_data(self, obj):
        return from_emcee(obj, var_names=["ln(f)", "b", "m"])

    def test__verify_var_names(self, obj):
        with pytest.raises(ValueError):
            from_emcee(obj, var_names=["not", "enough"])

    def test__verify_arg_names(self, obj):
        with pytest.raises(ValueError):
            from_emcee(obj, arg_names=["not", "enough"])


class TestPyMC3NetCDFUtils(BaseArvizTest):
    @classmethod
    def setup_class(cls):
        # Data of the Eight Schools Model
        cls.data = eight_schools_params()
        cls.draws, cls.chains = 500, 2
        cls.model, cls.obj = load_cached_models(cls.draws, cls.chains)["pymc3"]

    def get_inference_data(self):
        with self.model:
            prior = pm.sample_prior_predictive()
            posterior_predictive = pm.sample_posterior_predictive(self.obj)

        return from_pymc3(
            trace=self.obj,
            prior=prior,
            posterior_predictive=posterior_predictive,
            coords={"school": np.arange(self.data["J"])},
            dims={"theta": ["school"], "theta_tilde": ["school"]},
        )

    def test_sampler_stats(self):
        inference_data = self.get_inference_data()
        assert hasattr(inference_data, "sample_stats")

    def test_posterior_predictive(self):
        inference_data = self.get_inference_data()
        assert hasattr(inference_data, "posterior_predictive")

    def test_prior(self):
        inference_data = self.get_inference_data()
        assert hasattr(inference_data, "prior")


class TestPyStanNetCDFUtils(BaseArvizTest):
    @classmethod
    def setup_class(cls):
        # Data of the Eight Schools Model
        cls.data = eight_schools_params()
        cls.draws, cls.chains = 500, 2
        cls.model, cls.obj = load_cached_models(cls.draws, cls.chains)["pystan"]

    def get_inference_data(self):
        """log_likelihood as a var."""
        prior = pystan_extract_unpermuted(self.obj)
        prior = {"theta_test": prior["theta"]}
        return from_pystan(
            fit=self.obj,
            prior=prior,
            posterior_predictive="y_hat",
            observed_data=["y"],
            log_likelihood="log_lik",
            coords={"school": np.arange(self.data["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "theta_tilde": ["school"],
            },
        )

    def get_inference_data2(self):
        """log_likelihood as a ndarray."""
        # dictionary
        observed_data = {"y_hat": self.data["y"]}
        # ndarray
        log_likelihood = pystan_extract_normal(self.obj, "log_lik")["log_lik"]
        return from_pystan(
            fit=self.obj,
            posterior_predictive=["y_hat"],
            observed_data=observed_data,
            log_likelihood=log_likelihood,
            coords={"school": np.arange(self.data["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "theta_tilde": ["school"],
            },
        )

    def get_inference_data3(self):
        """log_likelihood as a ndarray."""
        # ndarray
        log_likelihood = pystan_extract_normal(self.obj, "log_lik")["log_lik"]
        return from_pystan(
            fit=self.obj,
            posterior_predictive=["y_hat"],
            observed_data=["y"],
            log_likelihood=log_likelihood,
            coords={"school": np.arange(self.data["J"])},
            dims={
                "theta": ["school"],
                "y": ["school"],
                "log_lik": ["school"],
                "y_hat": ["school"],
                "theta_tilde": ["school"],
            },
        )

    def test_sampler_stats(self):
        inference_data = self.get_inference_data()
        assert hasattr(inference_data, "sample_stats")

    def test_inference_data(self):
        inference_data1 = self.get_inference_data()
        inference_data2 = self.get_inference_data2()
        inference_data3 = self.get_inference_data3()
        assert hasattr(inference_data1.sample_stats, "log_likelihood")
        assert hasattr(inference_data1.prior, "theta_test")
        assert hasattr(inference_data1.observed_data, "y")
        assert hasattr(inference_data2.sample_stats, "log_likelihood")
        assert hasattr(inference_data2.observed_data, "y_hat")
        assert hasattr(inference_data3.sample_stats, "log_likelihood")
        assert hasattr(inference_data3.observed_data, "y")


class TestCmdStanNetCDFUtils(BaseArvizTest):

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

    def get_inference_data(self, output, **kwargs):
        return from_cmdstan(output=output, **kwargs)

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

            output --> str, list of str
            prior --> str, list of str
            posterior_predictive --> str, variable in output
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
                output=path,
                prior=path,
                posterior_predictive="y_hat",
                observed_data=observed_data_paths[0],
                observed_data_var="y",
                log_likelihood="log_lik",
                coords={"school": np.arange(8)},
                dims={
                    "theta": ["school"],
                    "y": ["school"],
                    "log_lik": ["school"],
                    "y_hat": ["school"],
                    "theta_tilde": ["school"],
                },
            )
            assert hasattr(inference_data, "posterior")
            assert hasattr(inference_data, "sample_stats")
            assert hasattr(inference_data.sample_stats, "log_likelihood")
            assert hasattr(inference_data, "posterior_predictive")
            assert hasattr(inference_data, "observed_data")

    def test_inference_data_input_types2(self, paths, observed_data_paths):
        """Check input types (change, see earlier)

            posterior_predictive --> List[str], variable in output
            observed_data_var --> List[str], variable
        """
        for key, path in paths.items():
            if "eight" not in key:
                continue
            inference_data = self.get_inference_data(
                output=path,
                prior=path,
                posterior_predictive=["y_hat"],
                observed_data=observed_data_paths[0],
                observed_data_var=["y"],
                log_likelihood="log_lik",
                coords={"school": np.arange(8)},
                dims={
                    "theta": ["school"],
                    "y": ["school"],
                    "log_lik": ["school"],
                    "y_hat": ["school"],
                    "theta_tilde": ["school"],
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
                output=path,
                prior=path,
                posterior_predictive=post_pred,
                observed_data=observed_data_paths[0],
                observed_data_var=["y"],
                log_likelihood="log_lik",
                coords={"school": np.arange(8), "log_lik_dim_0": np.arange(8)},
                dims={
                    "theta": ["school"],
                    "y": ["school"],
                    "y_hat": ["school"],
                    "theta_tilde": ["school"],
                },
            )
            assert hasattr(inference_data, "posterior")
            assert hasattr(inference_data, "sample_stats")
            assert hasattr(inference_data.sample_stats, "log_likelihood")
            assert hasattr(inference_data, "posterior_predictive")
            assert hasattr(inference_data, "observed_data")

    def test_inference_data_input_types4(self, paths, observed_data_paths):
        """Check input types (change, see earlier)

            coords --> one to many + one to one (non-default dim)
            dims --> one to many + one to one
        """
        for key, path in paths.items():
            if "eight" not in key:
                continue
            post_pred = paths["eight_schools"]
            inference_data = self.get_inference_data(
                output=path,
                prior=path,
                posterior_predictive=post_pred,
                observed_data=observed_data_paths[0],
                observed_data_var=["y"],
                log_likelihood=["log_lik"],
                coords={"school": np.arange(8), "log_lik_dim": np.arange(8)},
                dims={
                    "theta": ["school"],
                    "y": ["school"],
                    "log_lik": ["log_lik_dim"],
                    "y_hat": ["school"],
                    "theta_tilde": ["school"],
                },
            )
            assert hasattr(inference_data, "posterior")
            assert hasattr(inference_data, "sample_stats")
            assert hasattr(inference_data.sample_stats, "log_likelihood")
            assert hasattr(inference_data, "posterior_predictive")
            assert hasattr(inference_data, "observed_data")

    def test_inference_data_bad_csv(self, paths):
        """Check ValueError for csv with missing headers"""
        for key, paths in paths.items():
            if "missing" not in key:
                continue
            for path in paths:
                with pytest.raises(ValueError):
                    self.get_inference_data(output=path)

    def test_inference_data_observed_data1(self, observed_data_paths):
        """Read Rdump, check shapes are correct

            All variables
        """
        path = observed_data_paths[1]
        inference_data = self.get_inference_data(output=None, observed_data=path)
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
            output=None, observed_data=path, observed_data_var="x"
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
            output=None, observed_data=path, observed_data_var=["x"]
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
            output=None, observed_data=path, observed_data_var=["y", "Z"]
        )
        assert hasattr(inference_data, "observed_data")
        assert len(inference_data.observed_data.data_vars) == 2
        assert inference_data.observed_data["y"].shape == (3,)
        assert inference_data.observed_data["Z"].shape == (4, 5)
