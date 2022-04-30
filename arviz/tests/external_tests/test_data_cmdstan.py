# pylint: disable=no-member, invalid-name, redefined-outer-name
# pylint: disable=too-many-lines
import os

import numpy as np
import pytest

from ... import from_cmdstan

from ..helpers import check_multiple_attrs


class TestDataCmdStan:
    @pytest.fixture(scope="session")
    def data_directory(self):
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, "..", "saved_models")
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
            "eight_schools_glob": os.path.join(
                data_directory, "cmdstan/eight_schools_output[0-9].csv"
            ),
            "eight_schools": [
                os.path.join(data_directory, "cmdstan/eight_schools_output1.csv"),
                os.path.join(data_directory, "cmdstan/eight_schools_output2.csv"),
                os.path.join(data_directory, "cmdstan/eight_schools_output3.csv"),
                os.path.join(data_directory, "cmdstan/eight_schools_output4.csv"),
            ],
        }
        return paths

    @pytest.fixture(scope="class")
    def observed_data_paths(self, data_directory):
        observed_data_paths = [
            os.path.join(data_directory, "cmdstan/eight_schools.data.R"),
            os.path.join(data_directory, "cmdstan/example_stan.data.R"),
            os.path.join(data_directory, "cmdstan/example_stan.json"),
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
            assert "step_size" in inference_data.sample_stats.attrs
            assert inference_data.sample_stats.attrs["step_size"] == "stepsize"

    def test_inference_data_shapes(self, paths):
        """Assert that shapes are transformed correctly"""
        for key, path in paths.items():
            if "eight" in key or "missing" in key:
                continue
            inference_data = self.get_inference_data(path)
            test_dict = {"posterior": ["x", "y", "Z"]}
            fails = check_multiple_attrs(test_dict, inference_data)
            assert not fails
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
            assert "comments" in inference_data.posterior.attrs

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
                predictions="y_hat",
                prior=path,
                prior_predictive="y_hat",
                observed_data=observed_data_paths[0],
                observed_data_var="y",
                constant_data=observed_data_paths[0],
                constant_data_var="y",
                predictions_constant_data=observed_data_paths[0],
                predictions_constant_data_var="y",
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
            test_dict = {
                "posterior": ["mu", "tau", "theta_tilde", "theta"],
                "posterior_predictive": ["y_hat"],
                "predictions": ["y_hat"],
                "prior": ["mu", "tau", "theta_tilde", "theta"],
                "prior_predictive": ["y_hat"],
                "sample_stats": ["diverging"],
                "observed_data": ["y"],
                "constant_data": ["y"],
                "predictions_constant_data": ["y"],
                "log_likelihood": ["log_lik"],
            }
            if "output_warmup" in path:
                test_dict.update({"warmup_posterior": ["mu", "tau", "theta_tilde", "theta"]})
            fails = check_multiple_attrs(test_dict, inference_data)
            assert not fails

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
                predictions=["y_hat"],
                prior=path,
                prior_predictive=["y_hat"],
                observed_data=observed_data_paths[0],
                observed_data_var=["y"],
                constant_data=observed_data_paths[0],
                constant_data_var=["y"],
                predictions_constant_data=observed_data_paths[0],
                predictions_constant_data_var=["y"],
                coords={"school": np.arange(8)},
                dims={
                    "theta": ["school"],
                    "y": ["school"],
                    "log_lik": ["school"],
                    "y_hat": ["school"],
                    "eta": ["school"],
                },
                dtypes={"theta": np.int64},
            )
            test_dict = {
                "posterior": ["mu", "tau", "theta_tilde", "theta"],
                "posterior_predictive": ["y_hat"],
                "predictions": ["y_hat"],
                "prior": ["mu", "tau", "theta_tilde", "theta"],
                "prior_predictive": ["y_hat"],
                "sample_stats": ["diverging"],
                "observed_data": ["y"],
                "constant_data": ["y"],
                "predictions_constant_data": ["y"],
                "log_likelihood": ["log_lik"],
            }
            fails = check_multiple_attrs(test_dict, inference_data)
            assert not fails
            assert isinstance(inference_data.posterior.theta.data.flat[0], np.integer)

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
                log_likelihood=["log_lik", "y_hat"],
                coords={
                    "school": np.arange(8),
                    "log_lik_dim_0": np.arange(8),
                    "y_hat": np.arange(8),
                },
                dims={"theta": ["school"], "y": ["school"], "y_hat": ["school"], "eta": ["school"]},
            )
            test_dict = {
                "posterior": ["mu", "tau", "theta_tilde", "theta"],
                "sample_stats": ["diverging"],
                "prior": ["mu", "tau", "theta_tilde", "theta"],
                "prior_predictive": ["y_hat"],
                "observed_data": ["y"],
                "posterior_predictive": ["y_hat"],
                "log_likelihood": ["log_lik", "y_hat"],
            }
            fails = check_multiple_attrs(test_dict, inference_data)
            assert not fails

    def test_inference_data_input_types4(self, paths):
        """Check input types (change, see earlier)

        coords --> one to many + one to one (non-default dim)
        dims --> one to many + one to one
        """

        paths_ = paths["no_warmup"]
        for path in [paths_, paths_[0]]:
            inference_data = self.get_inference_data(
                posterior=path,
                posterior_predictive=path,
                prior=path,
                prior_predictive=path,
                observed_data=None,
                observed_data_var=None,
                log_likelihood=False,
                coords={"rand": np.arange(3)},
                dims={"x": ["rand"]},
            )
            test_dict = {
                "posterior": ["x", "y", "Z"],
                "prior": ["x", "y", "Z"],
                "prior_predictive": ["x", "y", "Z"],
                "sample_stats": ["lp"],
                "sample_stats_prior": ["lp"],
                "posterior_predictive": ["x", "y", "Z"],
                "~log_likelihood": [""],
            }
            fails = check_multiple_attrs(test_dict, inference_data)
            assert not fails

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
                log_likelihood=["y_hat"],
                coords={"school": np.arange(8), "log_lik_dim": np.arange(8)},
                dims={
                    "theta": ["school"],
                    "y": ["school"],
                    "log_lik": ["log_lik_dim"],
                    "y_hat": ["school"],
                    "eta": ["school"],
                },
            )
            test_dict = {
                "posterior": ["mu", "tau", "theta_tilde", "theta", "log_lik"],
                "prior": ["mu", "tau", "theta_tilde", "theta"],
                "log_likelihood": ["y_hat", "~log_lik"],
                "observed_data": ["y"],
                "sample_stats_prior": ["lp"],
            }
            fails = check_multiple_attrs(test_dict, inference_data)
            assert not fails

    def test_inference_data_input_types6(self, paths, observed_data_paths):
        """Check input types (change, see earlier)

        log_likelihood --> dict
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
                log_likelihood={"y": "log_lik"},
                coords={
                    "school": np.arange(8),
                    "log_lik_dim_0": np.arange(8),
                    "y_hat": np.arange(8),
                },
                dims={"theta": ["school"], "y": ["school"], "y_hat": ["school"], "eta": ["school"]},
            )
            test_dict = {
                "posterior": ["mu", "tau", "theta_tilde", "theta"],
                "sample_stats": ["diverging"],
                "prior": ["mu", "tau", "theta_tilde", "theta"],
                "prior_predictive": ["y_hat"],
                "observed_data": ["y"],
                "posterior_predictive": ["y_hat"],
                "log_likelihood": ["y", "~log_lik"],
            }
            fails = check_multiple_attrs(test_dict, inference_data)
            assert not fails

    def test_inference_data_observed_data1(self, observed_data_paths):
        """Read Rdump/JSON, check shapes are correct

        All variables
        """
        # Check the Rdump (idx=1) and equivalent JSON data file (idx=2)
        for data_idx in (1, 2):
            path = observed_data_paths[data_idx]
            inference_data = self.get_inference_data(posterior=None, observed_data=path)
            assert hasattr(inference_data, "observed_data")
            assert len(inference_data.observed_data.data_vars) == 3
            assert inference_data.observed_data["x"].shape == (1,)
            assert inference_data.observed_data["x"][0] == 1
            assert inference_data.observed_data["y"].shape == (3,)
            assert inference_data.observed_data["Z"].shape == (4, 5)

    def test_inference_data_observed_data2(self, observed_data_paths):
        """Read Rdump/JSON, check shapes are correct

        One variable as str
        """
        # Check the Rdump (idx=1) and equivalent JSON data file (idx=2)
        for data_idx in (1, 2):
            path = observed_data_paths[data_idx]
            inference_data = self.get_inference_data(
                posterior=None, observed_data=path, observed_data_var="x"
            )
            assert hasattr(inference_data, "observed_data")
            assert len(inference_data.observed_data.data_vars) == 1
            assert inference_data.observed_data["x"].shape == (1,)

    def test_inference_data_observed_data3(self, observed_data_paths):
        """Read Rdump/JSON, check shapes are correct

        One variable as a list
        """
        # Check the Rdump (idx=1) and equivalent JSON data file (idx=2)
        for data_idx in (1, 2):
            path = observed_data_paths[data_idx]
            inference_data = self.get_inference_data(
                posterior=None, observed_data=path, observed_data_var=["x"]
            )
            assert hasattr(inference_data, "observed_data")
            assert len(inference_data.observed_data.data_vars) == 1
            assert inference_data.observed_data["x"].shape == (1,)

    def test_inference_data_observed_data4(self, observed_data_paths):
        """Read Rdump/JSON, check shapes are correct

        Many variables as list
        """
        # Check the Rdump (idx=1) and equivalent JSON data file (idx=2)
        for data_idx in (1, 2):
            path = observed_data_paths[data_idx]
            inference_data = self.get_inference_data(
                posterior=None, observed_data=path, observed_data_var=["y", "Z"]
            )
            assert hasattr(inference_data, "observed_data")
            assert len(inference_data.observed_data.data_vars) == 2
            assert inference_data.observed_data["y"].shape == (3,)
            assert inference_data.observed_data["Z"].shape == (4, 5)
