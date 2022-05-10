# pylint: disable=too-many-lines
"""CmdStan-specific conversion code."""
try:
    import ujson as json
except ImportError:
    # Can't find ujson using json
    # mypy struggles with conditional imports expressed as catching ImportError:
    # https://github.com/python/mypy/issues/1153
    import json  # type: ignore
import logging
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .. import utils
from ..rcparams import rcParams
from .base import CoordSpec, DimSpec, dict_to_dataset, infer_stan_dtypes, requires
from .inference_data import InferenceData

_log = logging.getLogger(__name__)


def check_glob(path, group, disable_glob):
    """Find files with glob."""
    if isinstance(path, str) and (not disable_glob):
        path_glob = glob(path)
        if path_glob:
            path = sorted(path_glob)
            msg = "\n".join(f"{i}: {os.path.normpath(fpath)}" for i, fpath in enumerate(path, 1))
            len_p = len(path)
            _log.info("glob found %d files for '%s':\n%s", len_p, group, msg)
    return path


class CmdStanConverter:
    """Encapsulate CmdStan specific logic."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        *,
        posterior=None,
        posterior_predictive=None,
        predictions=None,
        prior=None,
        prior_predictive=None,
        observed_data=None,
        observed_data_var=None,
        constant_data=None,
        constant_data_var=None,
        predictions_constant_data=None,
        predictions_constant_data_var=None,
        log_likelihood=None,
        index_origin=None,
        coords=None,
        dims=None,
        disable_glob=False,
        save_warmup=None,
        dtypes=None,
    ):
        self.posterior_ = check_glob(posterior, "posterior", disable_glob)
        self.posterior_predictive = check_glob(
            posterior_predictive, "posterior_predictive", disable_glob
        )
        self.predictions = check_glob(predictions, "predictions", disable_glob)
        self.prior_ = check_glob(prior, "prior", disable_glob)
        self.prior_predictive = check_glob(prior_predictive, "prior_predictive", disable_glob)
        self.log_likelihood = check_glob(log_likelihood, "log_likelihood", disable_glob)
        self.observed_data = observed_data
        self.observed_data_var = observed_data_var
        self.constant_data = constant_data
        self.constant_data_var = constant_data_var
        self.predictions_constant_data = predictions_constant_data
        self.predictions_constant_data_var = predictions_constant_data_var
        self.coords = coords if coords is not None else {}
        self.dims = dims if dims is not None else {}

        self.posterior = None
        self.prior = None
        self.attrs = None
        self.attrs_prior = None

        self.save_warmup = rcParams["data.save_warmup"] if save_warmup is None else save_warmup
        self.index_origin = index_origin

        if dtypes is None:
            dtypes = {}
        elif isinstance(dtypes, str):
            dtypes_path = Path(dtypes)
            if dtypes_path.exists():
                with dtypes_path.open("r", encoding="UTF-8") as f_obj:
                    model_code = f_obj.read()
            else:
                model_code = dtypes

            dtypes = infer_stan_dtypes(model_code)

        self.dtypes = dtypes

        # populate posterior and sample_stats
        self._parse_posterior()
        self._parse_prior()

        if (
            self.log_likelihood is None
            and self.posterior_ is not None
            and any(name.split(".")[0] == "log_lik" for name in self.posterior_columns)
        ):
            self.log_likelihood = ["log_lik"]
        elif isinstance(self.log_likelihood, bool):
            self.log_likelihood = None

    @requires("posterior_")
    def _parse_posterior(self):
        """Read csv paths to list of ndarrays."""
        paths = self.posterior_
        if isinstance(paths, str):
            paths = [paths]

        chain_data = []
        columns = None
        for path in paths:
            output_data = _read_output(path)
            chain_data.append(output_data)
            if columns is None:
                columns = output_data

        self.posterior = (
            [item["sample"] for item in chain_data],
            [item["sample_warmup"] for item in chain_data],
        )
        self.posterior_columns = columns["sample_columns"]
        self.sample_stats_columns = columns["sample_stats_columns"]

        attrs = {}
        for item in chain_data:
            for key, value in item["configuration_info"].items():
                if key not in attrs:
                    attrs[key] = []
                attrs[key].append(value)
        self.attrs = attrs

    @requires("prior_")
    def _parse_prior(self):
        """Read csv paths to list of ndarrays."""
        paths = self.prior_
        if isinstance(paths, str):
            paths = [paths]

        chain_data = []
        columns = None
        for path in paths:
            output_data = _read_output(path)
            chain_data.append(output_data)
            if columns is None:
                columns = output_data

        self.prior = (
            [item["sample"] for item in chain_data],
            [item["sample_warmup"] for item in chain_data],
        )
        self.prior_columns = columns["sample_columns"]
        self.sample_stats_prior_columns = columns["sample_stats_columns"]

        attrs = {}
        for item in chain_data:
            for key, value in item["configuration_info"].items():
                if key not in attrs:
                    attrs[key] = []
                attrs[key].append(value)
        self.attrs_prior = attrs

    @requires("posterior")
    def posterior_to_xarray(self):
        """Extract posterior samples from output csv."""
        columns = self.posterior_columns

        # filter posterior_predictive, predictions and log_likelihood
        posterior_predictive = self.posterior_predictive
        if posterior_predictive is None or (
            isinstance(posterior_predictive, str) and posterior_predictive.lower().endswith(".csv")
        ):
            posterior_predictive = []
        elif isinstance(posterior_predictive, str):
            posterior_predictive = [
                col for col in columns if posterior_predictive == col.split(".")[0]
            ]
        else:
            posterior_predictive = [
                col
                for col in columns
                if any(item == col.split(".")[0] for item in posterior_predictive)
            ]

        predictions = self.predictions
        if predictions is None or (
            isinstance(predictions, str) and predictions.lower().endswith(".csv")
        ):
            predictions = []
        elif isinstance(predictions, str):
            predictions = [col for col in columns if predictions == col.split(".")[0]]
        else:
            predictions = [
                col for col in columns if any(item == col.split(".")[0] for item in predictions)
            ]

        log_likelihood = self.log_likelihood
        if log_likelihood is None or (
            isinstance(log_likelihood, str) and log_likelihood.lower().endswith(".csv")
        ):
            log_likelihood = []
        elif isinstance(log_likelihood, str):
            log_likelihood = [col for col in columns if log_likelihood == col.split(".")[0]]
        elif isinstance(log_likelihood, dict):
            log_likelihood = [
                col
                for col in columns
                if any(item == col.split(".")[0] for item in log_likelihood.values())
            ]
        else:
            log_likelihood = [
                col for col in columns if any(item == col.split(".")[0] for item in log_likelihood)
            ]

        invalid_cols = posterior_predictive + predictions + log_likelihood
        valid_cols = {col: idx for col, idx in columns.items() if col not in invalid_cols}
        data = _unpack_ndarrays(self.posterior[0], valid_cols, self.dtypes)
        data_warmup = _unpack_ndarrays(self.posterior[1], valid_cols, self.dtypes)
        return (
            dict_to_dataset(
                data,
                coords=self.coords,
                dims=self.dims,
                attrs=self.attrs,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                coords=self.coords,
                dims=self.dims,
                attrs=self.attrs,
                index_origin=self.index_origin,
            ),
        )

    @requires("posterior")
    @requires("sample_stats_columns")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from fit."""
        dtypes = {"diverging": bool, "n_steps": np.int64, "tree_depth": np.int64, **self.dtypes}
        rename_dict = {
            "divergent": "diverging",
            "n_leapfrog": "n_steps",
            "treedepth": "tree_depth",
            "stepsize": "step_size",
            "accept_stat": "acceptance_rate",
        }

        columns_new = {}
        for key, idx in self.sample_stats_columns.items():
            name = re.sub("__$", "", key)
            name = rename_dict.get(name, name)
            columns_new[name] = idx

        data = _unpack_ndarrays(self.posterior[0], columns_new, dtypes)
        data_warmup = _unpack_ndarrays(self.posterior[1], columns_new, dtypes)
        return (
            dict_to_dataset(
                data,
                coords=self.coords,
                dims=self.dims,
                attrs={item: key for key, item in rename_dict.items()},
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                coords=self.coords,
                dims=self.dims,
                attrs={item: key for key, item in rename_dict.items()},
                index_origin=self.index_origin,
            ),
        )

    @requires("posterior")
    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        posterior_predictive = self.posterior_predictive

        if (
            isinstance(posterior_predictive, (tuple, list))
            and posterior_predictive[0].endswith(".csv")
        ) or (isinstance(posterior_predictive, str) and posterior_predictive.endswith(".csv")):
            if isinstance(posterior_predictive, str):
                posterior_predictive = [posterior_predictive]
            chain_data = []
            chain_data_warmup = []
            columns = None
            attrs = {}
            for path in posterior_predictive:
                parsed_output = _read_output(path)
                chain_data.append(parsed_output["sample"])
                chain_data_warmup.append(parsed_output["sample_warmup"])
                if columns is None:
                    columns = parsed_output["sample_columns"]

                for key, value in parsed_output["configuration_info"].items():
                    if key not in attrs:
                        attrs[key] = []
                    attrs[key].append(value)

            data = _unpack_ndarrays(chain_data, columns, self.dtypes)
            data_warmup = _unpack_ndarrays(chain_data_warmup, columns, self.dtypes)

        else:
            if isinstance(posterior_predictive, str):
                posterior_predictive = [posterior_predictive]
            columns = {
                col: idx
                for col, idx in self.posterior_columns.items()
                if any(item == col.split(".")[0] for item in posterior_predictive)
            }
            data = _unpack_ndarrays(self.posterior[0], columns, self.dtypes)
            data_warmup = _unpack_ndarrays(self.posterior[1], columns, self.dtypes)

            attrs = None
        return (
            dict_to_dataset(
                data,
                coords=self.coords,
                dims=self.dims,
                attrs=attrs,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                coords=self.coords,
                dims=self.dims,
                attrs=attrs,
                index_origin=self.index_origin,
            ),
        )

    @requires("posterior")
    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert out of sample predictions samples to xarray."""
        predictions = self.predictions

        if (isinstance(predictions, (tuple, list)) and predictions[0].endswith(".csv")) or (
            isinstance(predictions, str) and predictions.endswith(".csv")
        ):
            if isinstance(predictions, str):
                predictions = [predictions]
            chain_data = []
            chain_data_warmup = []
            columns = None
            attrs = {}
            for path in predictions:
                parsed_output = _read_output(path)
                chain_data.append(parsed_output["sample"])
                chain_data_warmup.append(parsed_output["sample_warmup"])
                if columns is None:
                    columns = parsed_output["sample_columns"]

                for key, value in parsed_output["configuration_info"].items():
                    if key not in attrs:
                        attrs[key] = []
                    attrs[key].append(value)

            data = _unpack_ndarrays(chain_data, columns, self.dtypes)
            data_warmup = _unpack_ndarrays(chain_data_warmup, columns, self.dtypes)
        else:
            if isinstance(predictions, str):
                predictions = [predictions]
            columns = {
                col: idx
                for col, idx in self.posterior_columns.items()
                if any(item == col.split(".")[0] for item in predictions)
            }
            data = _unpack_ndarrays(self.posterior[0], columns, self.dtypes)
            data_warmup = _unpack_ndarrays(self.posterior[1], columns, self.dtypes)

            attrs = None
        return (
            dict_to_dataset(
                data,
                coords=self.coords,
                dims=self.dims,
                attrs=attrs,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                coords=self.coords,
                dims=self.dims,
                attrs=attrs,
                index_origin=self.index_origin,
            ),
        )

    @requires("posterior")
    @requires("log_likelihood")
    def log_likelihood_to_xarray(self):
        """Convert elementwise log_likelihood samples to xarray."""
        log_likelihood = self.log_likelihood

        if (isinstance(log_likelihood, (tuple, list)) and log_likelihood[0].endswith(".csv")) or (
            isinstance(log_likelihood, str) and log_likelihood.endswith(".csv")
        ):
            if isinstance(log_likelihood, str):
                log_likelihood = [log_likelihood]

            chain_data = []
            chain_data_warmup = []
            columns = None
            attrs = {}
            for path in log_likelihood:
                parsed_output = _read_output(path)
                chain_data.append(parsed_output["sample"])
                chain_data_warmup.append(parsed_output["sample_warmup"])

                if columns is None:
                    columns = parsed_output["sample_columns"]

                for key, value in parsed_output["configuration_info"].items():
                    if key not in attrs:
                        attrs[key] = []
                    attrs[key].append(value)
            data = _unpack_ndarrays(chain_data, columns, self.dtypes)
            data_warmup = _unpack_ndarrays(chain_data_warmup, columns, self.dtypes)
        else:
            if isinstance(log_likelihood, dict):
                log_lik_to_obs_name = {v: k for k, v in log_likelihood.items()}
                columns = {
                    col.replace(col_name, log_lik_to_obs_name[col_name]): idx
                    for col, col_name, idx in (
                        (col, col.split(".")[0], idx) for col, idx in self.posterior_columns.items()
                    )
                    if any(item == col_name for item in log_likelihood.values())
                }
            else:
                if isinstance(log_likelihood, str):
                    log_likelihood = [log_likelihood]
                columns = {
                    col: idx
                    for col, idx in self.posterior_columns.items()
                    if any(item == col.split(".")[0] for item in log_likelihood)
                }
            data = _unpack_ndarrays(self.posterior[0], columns, self.dtypes)
            data_warmup = _unpack_ndarrays(self.posterior[1], columns, self.dtypes)
            attrs = None
        return (
            dict_to_dataset(
                data,
                coords=self.coords,
                dims=self.dims,
                attrs=attrs,
                index_origin=self.index_origin,
                skip_event_dims=True,
            ),
            dict_to_dataset(
                data_warmup,
                coords=self.coords,
                dims=self.dims,
                attrs=attrs,
                index_origin=self.index_origin,
                skip_event_dims=True,
            ),
        )

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        # filter prior_predictive
        prior_predictive = self.prior_predictive

        columns = self.prior_columns

        if prior_predictive is None or (
            isinstance(prior_predictive, str) and prior_predictive.lower().endswith(".csv")
        ):
            prior_predictive = []
        elif isinstance(prior_predictive, str):
            prior_predictive = [col for col in columns if prior_predictive == col.split(".")[0]]
        else:
            prior_predictive = [
                col
                for col in columns
                if any(item == col.split(".")[0] for item in prior_predictive)
            ]

        invalid_cols = prior_predictive
        valid_cols = {col: idx for col, idx in columns.items() if col not in invalid_cols}
        data = _unpack_ndarrays(self.prior[0], valid_cols, self.dtypes)
        data_warmup = _unpack_ndarrays(self.prior[1], valid_cols, self.dtypes)
        return (
            dict_to_dataset(
                data,
                coords=self.coords,
                dims=self.dims,
                attrs=self.attrs_prior,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                coords=self.coords,
                dims=self.dims,
                attrs=self.attrs_prior,
                index_origin=self.index_origin,
            ),
        )

    @requires("prior")
    @requires("sample_stats_prior_columns")
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats from fit."""
        dtypes = {"diverging": bool, "n_steps": np.int64, "tree_depth": np.int64, **self.dtypes}
        rename_dict = {
            "divergent": "diverging",
            "n_leapfrog": "n_steps",
            "treedepth": "tree_depth",
            "stepsize": "step_size",
            "accept_stat": "acceptance_rate",
        }

        columns_new = {}
        for key, idx in self.sample_stats_prior_columns.items():
            name = re.sub("__$", "", key)
            name = rename_dict.get(name, name)
            columns_new[name] = idx

        data = _unpack_ndarrays(self.posterior[0], columns_new, dtypes)
        data_warmup = _unpack_ndarrays(self.posterior[1], columns_new, dtypes)
        return (
            dict_to_dataset(
                data,
                coords=self.coords,
                dims=self.dims,
                attrs={item: key for key, item in rename_dict.items()},
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                coords=self.coords,
                dims=self.dims,
                attrs={item: key for key, item in rename_dict.items()},
                index_origin=self.index_origin,
            ),
        )

    @requires("prior")
    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        prior_predictive = self.prior_predictive

        if (
            isinstance(prior_predictive, (tuple, list)) and prior_predictive[0].endswith(".csv")
        ) or (isinstance(prior_predictive, str) and prior_predictive.endswith(".csv")):
            if isinstance(prior_predictive, str):
                prior_predictive = [prior_predictive]
            chain_data = []
            chain_data_warmup = []
            columns = None
            attrs = {}
            for path in prior_predictive:
                parsed_output = _read_output(path)
                chain_data.append(parsed_output["sample"])
                chain_data_warmup.append(parsed_output["sample_warmup"])
                if columns is None:
                    columns = parsed_output["sample_columns"]
                for key, value in parsed_output["configuration_info"].items():
                    if key not in attrs:
                        attrs[key] = []
                    attrs[key].append(value)
            data = _unpack_ndarrays(chain_data, columns, self.dtypes)
            data_warmup = _unpack_ndarrays(chain_data_warmup, columns, self.dtypes)
        else:
            if isinstance(prior_predictive, str):
                prior_predictive = [prior_predictive]
            columns = {
                col: idx
                for col, idx in self.prior_columns.items()
                if any(item == col.split(".")[0] for item in prior_predictive)
            }
            data = _unpack_ndarrays(self.prior[0], columns, self.dtypes)
            data_warmup = _unpack_ndarrays(self.prior[1], columns, self.dtypes)
            attrs = None
        return (
            dict_to_dataset(
                data,
                coords=self.coords,
                dims=self.dims,
                attrs=attrs,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                coords=self.coords,
                dims=self.dims,
                attrs=attrs,
                index_origin=self.index_origin,
            ),
        )

    @requires("observed_data")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        observed_data_raw = _read_data(self.observed_data)
        variables = self.observed_data_var
        if isinstance(variables, str):
            variables = [variables]
        observed_data = {
            key: utils.one_de(vals)
            for key, vals in observed_data_raw.items()
            if variables is None or key in variables
        }
        return dict_to_dataset(
            observed_data,
            coords=self.coords,
            dims=self.dims,
            default_dims=[],
            index_origin=self.index_origin,
        )

    @requires("constant_data")
    def constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        constant_data_raw = _read_data(self.constant_data)
        variables = self.constant_data_var
        if isinstance(variables, str):
            variables = [variables]
        constant_data = {
            key: utils.one_de(vals)
            for key, vals in constant_data_raw.items()
            if variables is None or key in variables
        }
        return dict_to_dataset(
            constant_data,
            coords=self.coords,
            dims=self.dims,
            default_dims=[],
            index_origin=self.index_origin,
        )

    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert predictions constant data to xarray."""
        predictions_constant_data_raw = _read_data(self.predictions_constant_data)
        variables = self.predictions_constant_data_var
        if isinstance(variables, str):
            variables = [variables]
        predictions_constant_data = {}
        for key, vals in predictions_constant_data_raw.items():
            if variables is not None and key not in variables:
                continue
            vals = utils.one_de(vals)
            predictions_constant_data[key] = utils.one_de(vals)
        return dict_to_dataset(
            predictions_constant_data,
            coords=self.coords,
            dims=self.dims,
            default_dims=[],
            index_origin=self.index_origin,
        )

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `output`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        return InferenceData(
            save_warmup=self.save_warmup,
            **{
                "posterior": self.posterior_to_xarray(),
                "sample_stats": self.sample_stats_to_xarray(),
                "log_likelihood": self.log_likelihood_to_xarray(),
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "prior": self.prior_to_xarray(),
                "sample_stats_prior": self.sample_stats_prior_to_xarray(),
                "prior_predictive": self.prior_predictive_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
                "constant_data": self.constant_data_to_xarray(),
                "predictions": self.predictions_to_xarray(),
                "predictions_constant_data": self.predictions_constant_data_to_xarray(),
            },
        )


def _process_configuration(comments):
    """Extract sampling information."""
    results = {
        "comments": "\n".join(comments),
        "stan_version": {},
    }

    comments_gen = iter(comments)

    for comment in comments_gen:
        comment = re.sub(r"^\s*#\s*|\s*\(Default\)\s*$", "", comment).strip()
        if comment.startswith("stan_version_"):
            key, val = re.sub(r"^\s*stan_version_", "", comment).split("=")
            results["stan_version"][key.strip()] = val.strip()
        elif comment.startswith("Step size"):
            _, val = comment.split("=")
            results["step_size"] = float(val.strip())
        elif "inverse mass matrix" in comment:
            comment = re.sub(r"^\s*#\s*", "", next(comments_gen)).strip()
            results["inverse_mass_matrix"] = [float(item) for item in comment.split(",")]
        elif ("seconds" in comment) and any(
            item in comment for item in ("(Warm-up)", "(Sampling)", "(Total)")
        ):
            value = re.sub(
                (
                    r"^Elapsed\s*Time:\s*|"
                    r"\s*seconds\s*\(Warm-up\)\s*|"
                    r"\s*seconds\s*\(Sampling\)\s*|"
                    r"\s*seconds\s*\(Total\)\s*"
                ),
                "",
                comment,
            )
            key = (
                "warmup_time_seconds"
                if "(Warm-up)" in comment
                else "sampling_time_seconds"
                if "(Sampling)" in comment
                else "total_time_seconds"
            )
            results[key] = float(value)
        elif "=" in comment:
            match_int = re.search(r"^(\S+)\s*=\s*([-+]?[0-9]+)$", comment)
            match_float = re.search(r"^(\S+)\s*=\s*([-+]?[0-9]+\.[0-9]+)$", comment)
            match_str = re.search(r"^(\S+)\s*=\s*(\S+)$", comment)
            match_empty = re.search(r"^(\S+)\s*=\s*$", comment)
            if match_int:
                key, value = match_int.group(1), match_int.group(2)
                results[key] = int(value)
            elif match_float:
                key, value = match_float.group(1), match_float.group(2)
                results[key] = float(value)
            elif match_str:
                key, value = match_str.group(1), match_str.group(2)
                results[key] = value
            elif match_empty:
                key = match_empty.group(1)
                results[key] = None

    results = {key: str(results[key]) for key in sorted(results)}
    return results


def _read_output_file(path):
    """Read Stan csv file to ndarray."""
    comments = []
    data = []
    columns = None
    with open(path, "rb") as f_obj:
        # read header
        for line in f_obj:
            if line.startswith(b"#"):
                comments.append(line.strip().decode("utf-8"))
                continue
            columns = {key: idx for idx, key in enumerate(line.strip().decode("utf-8").split(","))}
            break
        # read data
        for line in f_obj:
            line = line.strip()
            if line.startswith(b"#"):
                comments.append(line.decode("utf-8"))
                continue
            if line:
                data.append(np.array(line.split(b","), dtype=np.float64))

    return columns, np.array(data, dtype=np.float64), comments


def _read_output(path):
    """Read CmdStan output csv file.

    Parameters
    ----------
    path : str

    Returns
    -------
    Dict[str, Any]
    """
    # Read data
    columns, data, comments = _read_output_file(path)

    pconf = _process_configuration(comments)

    # split dataframe to warmup and draws
    saved_warmup = (
        int(pconf.get("save_warmup", 0))
        * int(pconf.get("num_warmup", 0))
        // int(pconf.get("thin", 1))
    )

    data_warmup = data[:saved_warmup]
    data = data[saved_warmup:]

    # Split data to sample_stats and sample
    sample_stats_columns = {col: idx for col, idx in columns.items() if col.endswith("__")}
    sample_columns = {col: idx for col, idx in columns.items() if col not in sample_stats_columns}

    return {
        "sample": data,
        "sample_warmup": data_warmup,
        "sample_columns": sample_columns,
        "sample_stats_columns": sample_stats_columns,
        "configuration_info": pconf,
    }


def _process_data_var(string):
    """Transform datastring to key, values pair.

    All values are transformed to floating point values.

    Parameters
    ----------
    string : str

    Returns
    -------
    Tuple[Str, Str]
        key, values pair
    """
    key, var = string.split("<-")
    if "structure" in var:
        var, dim = var.replace("structure(", "").replace(",", "").split(".Dim")
        # dtype = int if '.' not in var and 'e' not in var.lower() else float
        dtype = float
        var = var.replace("c(", "").replace(")", "").strip().split()
        dim = dim.replace("=", "").replace("c(", "").replace(")", "").strip().split()
        dim = tuple(map(int, dim))
        var = np.fromiter(map(dtype, var), dtype).reshape(dim, order="F")
    elif "c(" in var:
        # dtype = int if '.' not in var and 'e' not in var.lower() else float
        dtype = float
        var = var.replace("c(", "").replace(")", "").split(",")
        var = np.fromiter(map(dtype, var), dtype)
    else:
        # dtype = int if '.' not in var and 'e' not in var.lower() else float
        dtype = float
        var = dtype(var)
    return key.strip(), var


def _read_data(path):
    """Read Rdump or JSON output to dictionary.

    Parameters
    ----------
    path : str

    Returns
    -------
    Dict
        key, values pairs from Rdump/JSON formatted data.
    """
    data = {}
    with open(path, "r", encoding="utf8") as f_obj:
        if path.lower().endswith(".json"):
            return json.load(f_obj)
        var = ""
        for line in f_obj:
            if "<-" in line:
                if len(var):
                    key, var = _process_data_var(var)
                    data[key] = var
                var = ""
            var += f" {line.strip()}"
        if len(var):
            key, var = _process_data_var(var)
            data[key] = var
    return data


def _unpack_ndarrays(arrays, columns, dtypes=None):
    """Transform a list of ndarrays to dictionary containing ndarrays.

    Parameters
    ----------
    arrays : List[np.ndarray]
    columns: Dict[str, int]
    dtypes: Dict[str, Any]

    Returns
    -------
    Dict
        key, values pairs. Values are formatted to shape = (nchain, ndraws, *shape)
    """
    col_groups = defaultdict(list)
    for col, col_idx in columns.items():
        key, *loc = col.split(".")
        loc = tuple(int(i) - 1 for i in loc)
        col_groups[key].append((col_idx, loc))

    chains = len(arrays)
    draws = len(arrays[0])
    sample = {}
    if draws:
        for key, cols_locs in col_groups.items():
            ndim = np.array([loc for _, loc in cols_locs]).max(0) + 1
            dtype = dtypes.get(key, np.float64)
            sample[key] = np.zeros((chains, draws, *ndim), dtype=dtype)
            for col, loc in cols_locs:
                for chain_id, arr in enumerate(arrays):
                    draw = arr[:, col]
                    if loc == ():
                        sample[key][chain_id, :] = draw
                    else:
                        axis1_all = range(sample[key].shape[1])
                        slicer = (chain_id, axis1_all, *loc)
                        sample[key][slicer] = draw
    return sample


def from_cmdstan(
    posterior: Optional[Union[str, List[str]]] = None,
    *,
    posterior_predictive: Optional[Union[str, List[str]]] = None,
    predictions: Optional[Union[str, List[str]]] = None,
    prior: Optional[Union[str, List[str]]] = None,
    prior_predictive: Optional[Union[str, List[str]]] = None,
    observed_data: Optional[str] = None,
    observed_data_var: Optional[Union[str, List[str]]] = None,
    constant_data: Optional[str] = None,
    constant_data_var: Optional[Union[str, List[str]]] = None,
    predictions_constant_data: Optional[str] = None,
    predictions_constant_data_var: Optional[Union[str, List[str]]] = None,
    log_likelihood: Optional[Union[str, List[str]]] = None,
    index_origin: Optional[int] = None,
    coords: Optional[CoordSpec] = None,
    dims: Optional[DimSpec] = None,
    disable_glob: Optional[bool] = False,
    save_warmup: Optional[bool] = None,
    dtypes: Optional[Dict] = None,
) -> InferenceData:
    """Convert CmdStan data into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_cmdstan <creating_InferenceData>`

    Parameters
    ----------
    posterior : str or list of str, optional
        List of paths to output.csv files.
    posterior_predictive : str or list of str, optional
        Posterior predictive samples for the fit. If endswith ".csv" assumes file.
    predictions : str or list of str, optional
        Out of sample predictions samples for the fit. If endswith ".csv" assumes file.
    prior : str or list of str, optional
        List of paths to output.csv files
    prior_predictive : str or list of str, optional
        Prior predictive samples for the fit. If endswith ".csv" assumes file.
    observed_data : str, optional
        Observed data used in the sampling. Path to data file in Rdump or JSON format.
    observed_data_var : str or list of str, optional
        Variable(s) used for slicing observed_data. If not defined, all
        data variables are imported.
    constant_data : str, optional
        Constant data used in the sampling. Path to data file in Rdump or JSON format.
    constant_data_var : str or list of str, optional
        Variable(s) used for slicing constant_data. If not defined, all
        data variables are imported.
    predictions_constant_data : str, optional
        Constant data for predictions used in the sampling.
        Path to data file in Rdump or JSON format.
    predictions_constant_data_var : str or list of str, optional
        Variable(s) used for slicing predictions_constant_data.
        If not defined, all data variables are imported.
    log_likelihood : dict of {str: str}, list of str or str, optional
        Pointwise log_likelihood for the data. log_likelihood is extracted from the
        posterior. It is recommended to use this argument as a dictionary whose keys
        are observed variable names and its values are the variables storing log
        likelihood arrays in the Stan code. In other cases, a dictionary with keys
        equal to its values is used. By default, if a variable ``log_lik`` is
        present in the Stan model, it will be retrieved as pointwise log
        likelihood values. Use ``False`` to avoid this behaviour.
    index_origin : int, optional
        Starting value of integer coordinate values. Defaults to the value in rcParam
        ``data.index_origin``.
    coords : dict of {str: array_like}, optional
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict of {str: list of str}, optional
        A mapping from variables to a list of coordinate names for the variable.
    disable_glob : bool
        Don't use glob for string input. This means that all string input is
        assumed to be variable names (samples) or a path (data).
    save_warmup : bool
        Save warmup iterations into InferenceData object, if found in the input files.
        If not defined, use default defined by the rcParams.
    dtypes : dict or str
        A dictionary containing dtype information (int, float) for parameters.
        If input is a string, it is assumed to be a model code or path to model code file.

    Returns
    -------
    InferenceData object
    """
    return CmdStanConverter(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        predictions=predictions,
        prior=prior,
        prior_predictive=prior_predictive,
        observed_data=observed_data,
        observed_data_var=observed_data_var,
        constant_data=constant_data,
        constant_data_var=constant_data_var,
        predictions_constant_data=predictions_constant_data,
        predictions_constant_data_var=predictions_constant_data_var,
        log_likelihood=log_likelihood,
        index_origin=index_origin,
        coords=coords,
        dims=dims,
        disable_glob=disable_glob,
        save_warmup=save_warmup,
        dtypes=dtypes,
    ).to_inference_data()
