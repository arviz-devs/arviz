"""CmdStan-specific conversion code."""
from collections import defaultdict
from copy import deepcopy
from glob import glob
from typing import Optional, Union, List
import linecache
import os
import logging
import re

import numpy as np
import pandas as pd
import xarray as xr

from .. import utils
from .inference_data import InferenceData
from .base import requires, dict_to_dataset, generate_dims_coords, CoordSpec, DimSpec

_log = logging.getLogger(__name__)


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
        coords=None,
        dims=None
    ):
        if isinstance(posterior, str):
            posterior_glob = glob(posterior)
            if len(posterior_glob) > 1:
                posterior = sorted(posterior_glob)
                msg = "\n".join(
                    "{}: {}".format(i, os.path.normpath(path))
                    for i, path in enumerate(posterior, 1)
                )
                len_p = len(posterior)
                _log.info("glob found %d files for 'posterior':\n%s", len_p, msg)
        self.posterior_ = posterior
        if isinstance(posterior_predictive, str):
            posterior_predictive_glob = glob(posterior_predictive)
            if len(posterior_predictive_glob) > 1:
                posterior_predictive = sorted(posterior_predictive_glob)
                msg = "\n".join(
                    "{}: {}".format(i, os.path.normpath(path))
                    for i, path in enumerate(posterior_predictive, 1)
                )
                len_pp = len(posterior_predictive)
                _log.info("glob found %d files for 'posterior_predictive':\n%s", len_pp, msg)
        if isinstance(predictions, str):
            predictions_glob = glob(predictions)
            if len(predictions_glob) > 1:
                predictions = sorted(predictions_glob)
                msg = "\n".join(
                    "{}: {}".format(i, os.path.normpath(path))
                    for i, path in enumerate(predictions, 1)
                )
                len_p = len(predictions)
                _log.info("glob found %d files for 'predictions':\n%s", len_p, msg)
        if isinstance(prior, str):
            prior_glob = glob(prior)
            if len(prior_glob) > 1:
                prior = sorted(prior_glob)
                msg = "\n".join(
                    "{}: {}".format(i, os.path.normpath(path)) for i, path in enumerate(prior, 1)
                )
                len_p = len(prior)
                _log.info("glob found %d files for 'prior':\n%s", len_p, msg)
        if isinstance(prior_predictive, str):
            prior_predictive_glob = glob(prior_predictive)
            if len(prior_predictive_glob) > 1:
                prior_predictive = sorted(prior_predictive_glob)
                msg = "\n".join(
                    "{}: {}".format(i, os.path.normpath(path))
                    for i, path in enumerate(prior_predictive, 1)
                )
                len_pp = len(prior_predictive)
                _log.info("glob found %d files for 'prior_predictive':\n%s", len_pp, msg)
        if isinstance(log_likelihood, str):
            log_likelihood_glob = glob(log_likelihood)
            if len(log_likelihood_glob) > 1:
                log_likelihood = sorted(log_likelihood_glob)
                msg = "\n".join(
                    "{}: {}".format(i, os.path.normpath(path))
                    for i, path in enumerate(log_likelihood, 1)
                )
                len_ll = len(log_likelihood)
                _log.info("glob found %d files for 'log_likelihood':\n%s", len_ll, msg)
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions
        self.prior_ = prior
        self.prior_predictive = prior_predictive
        self.observed_data = observed_data
        self.observed_data_var = observed_data_var
        self.constant_data = constant_data
        self.constant_data_var = constant_data_var
        self.predictions_constant_data = predictions_constant_data
        self.predictions_constant_data_var = predictions_constant_data_var
        self.log_likelihood = log_likelihood
        self.coords = coords if coords is not None else {}
        self.dims = dims if dims is not None else {}
        self.posterior = None
        self.sample_stats = None
        self.prior = None
        self.sample_stats_prior = None

        # populate posterior and sample_stats
        self._parse_posterior()
        self._parse_prior()

    @requires("posterior_")
    def _parse_posterior(self):
        """Read csv paths to list of dataframes."""
        paths = self.posterior_
        if isinstance(paths, str):
            paths = [paths]
        chain_data = []
        for path in paths:
            parsed_output = _read_output(path)
            for sample, sample_stats, config, adaptation, timing in parsed_output:
                chain_data.append(
                    {
                        "sample": sample,
                        "sample_stats": sample_stats,
                        "configuration_info": config,
                        "adaptation_info": adaptation,
                        "timing_info": timing,
                    }
                )
        self.posterior = [item["sample"] for item in chain_data]
        self.sample_stats = [item["sample_stats"] for item in chain_data]

    @requires("prior_")
    def _parse_prior(self):
        """Read csv paths to list of dataframes."""
        paths = self.prior_
        if isinstance(paths, str):
            paths = [paths]
        chain_data = []
        for path in paths:
            parsed_output = _read_output(path)
            for sample, sample_stats, config, adaptation, timing in parsed_output:
                chain_data.append(
                    {
                        "sample": sample,
                        "sample_stats": sample_stats,
                        "configuration_info": config,
                        "adaptation_info": adaptation,
                        "timing_info": timing,
                    }
                )
        self.prior = [item["sample"] for item in chain_data]
        self.sample_stats_prior = [item["sample_stats"] for item in chain_data]

    @requires("posterior")
    def posterior_to_xarray(self):
        """Extract posterior samples from output csv."""
        columns = self.posterior[0].columns

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
        else:
            log_likelihood = [
                col for col in columns if any(item == col.split(".")[0] for item in log_likelihood)
            ]

        invalid_cols = posterior_predictive + predictions + log_likelihood
        valid_cols = [col for col in columns if col not in invalid_cols]
        data = _unpack_dataframes([item[valid_cols] for item in self.posterior])
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires("posterior")
    @requires("sample_stats")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from fit."""
        dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64}

        sampler_params = self.sample_stats

        for j, s_params in enumerate(sampler_params):
            rename_dict = {}
            for key in s_params:
                key_, *end = key.split(".")
                name = re.sub("__$", "", key_)
                name = "diverging" if name == "divergent" else name
                rename_dict[key] = ".".join((name, *end))
                sampler_params[j][key] = s_params[key].astype(dtypes.get(key_))
            sampler_params[j] = sampler_params[j].rename(columns=rename_dict)
        data = _unpack_dataframes(sampler_params)
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires("posterior")
    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        posterior_predictive = self.posterior_predictive
        columns = self.posterior[0].columns
        if (
            isinstance(posterior_predictive, (tuple, list))
            and posterior_predictive[0].endswith(".csv")
        ) or (isinstance(posterior_predictive, str) and posterior_predictive.endswith(".csv")):
            if isinstance(posterior_predictive, str):
                posterior_predictive = [posterior_predictive]
            chain_data = []
            for path in posterior_predictive:
                parsed_output = _read_output(path)
                for sample, *_ in parsed_output:
                    chain_data.append(sample)
            data = _unpack_dataframes(chain_data)
        else:
            if isinstance(posterior_predictive, str):
                posterior_predictive = [posterior_predictive]
            posterior_predictive_cols = [
                col
                for col in columns
                if any(item == col.split(".")[0] for item in posterior_predictive)
            ]
            data = _unpack_dataframes([item[posterior_predictive_cols] for item in self.posterior])
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires("posterior")
    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert out of sample predictions samples to xarray."""
        predictions = self.predictions
        columns = self.posterior[0].columns
        if (isinstance(predictions, (tuple, list)) and predictions[0].endswith(".csv")) or (
            isinstance(predictions, str) and predictions.endswith(".csv")
        ):
            if isinstance(predictions, str):
                predictions = [predictions]
            chain_data = []
            for path in predictions:
                parsed_output = _read_output(path)
                for sample, *_ in parsed_output:
                    chain_data.append(sample)
            data = _unpack_dataframes(chain_data)
        else:
            if isinstance(predictions, str):
                predictions = [predictions]
            predictions_cols = [
                col for col in columns if any(item == col.split(".")[0] for item in predictions)
            ]
            data = _unpack_dataframes([item[predictions_cols] for item in self.posterior])
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        # filter prior_predictive
        prior_predictive = self.prior_predictive
        columns = self.prior[0].columns
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
        valid_cols = [col for col in columns if col not in invalid_cols]
        data = _unpack_dataframes([item[valid_cols] for item in self.prior])
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires("prior")
    @requires("sample_stats_prior")
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats from fit."""
        dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64}

        # copy dims and coords
        dims = deepcopy(self.dims) if self.dims is not None else {}
        coords = deepcopy(self.coords) if self.coords is not None else {}

        sampler_params = self.sample_stats_prior
        for j, s_params in enumerate(sampler_params):
            rename_dict = {}
            for key in s_params:
                key_, *end = key.split(".")
                name = re.sub("__$", "", key_)
                name = "diverging" if name == "divergent" else name
                rename_dict[key] = ".".join((name, *end))
                sampler_params[j][key] = s_params[key].astype(dtypes.get(key_))
            sampler_params[j] = sampler_params[j].rename(columns=rename_dict)
        data = _unpack_dataframes(sampler_params)
        return dict_to_dataset(data, coords=coords, dims=dims)

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
            for path in prior_predictive:
                parsed_output = _read_output(path)
                for sample, *_ in parsed_output:
                    chain_data.append(sample)
            data = _unpack_dataframes(chain_data)
        else:
            if isinstance(prior_predictive, str):
                prior_predictive = [prior_predictive]
            prior_predictive_cols = [
                col
                for col in self.prior[0].columns
                if any(item == col.split(".")[0] for item in prior_predictive)
            ]
            data = _unpack_dataframes([item[prior_predictive_cols] for item in self.prior])
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires("observed_data")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        observed_data_raw = _read_data(self.observed_data)
        variables = self.observed_data_var
        if isinstance(variables, str):
            variables = [variables]
        observed_data = {}
        for key, vals in observed_data_raw.items():
            if variables is not None and key not in variables:
                continue
            vals = utils.one_de(vals)
            val_dims = self.dims.get(key)
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            observed_data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=observed_data)

    @requires("constant_data")
    def constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        constant_data_raw = _read_data(self.constant_data)
        variables = self.constant_data_var
        if isinstance(variables, str):
            variables = [variables]
        constant_data = {}
        for key, vals in constant_data_raw.items():
            if variables is not None and key not in variables:
                continue
            vals = utils.one_de(vals)
            val_dims = self.dims.get(key)
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            constant_data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=constant_data)

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
            val_dims = self.dims.get(key)
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            predictions_constant_data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=predictions_constant_data)

    @requires("posterior")
    @requires("log_likelihood")
    def log_likelihood_to_xarray(self):
        """Convert elementwise log_likelihood samples to xarray."""
        log_likelihood = self.log_likelihood
        columns = self.posterior[0].columns
        if (isinstance(log_likelihood, (tuple, list)) and log_likelihood[0].endswith(".csv")) or (
            isinstance(log_likelihood, str) and log_likelihood.endswith(".csv")
        ):
            if isinstance(log_likelihood, str):
                log_likelihood = [log_likelihood]
            chain_data = []
            for path in log_likelihood:
                parsed_output = _read_output(path)
                for sample, *_ in parsed_output:
                    chain_data.append(sample)
            data = _unpack_dataframes(chain_data)
        else:
            if isinstance(log_likelihood, str):
                log_likelihood = [log_likelihood]
            log_likelihood_cols = [
                col for col in columns if any(item == col.split(".")[0] for item in log_likelihood)
            ]
            data = _unpack_dataframes([item[log_likelihood_cols] for item in self.posterior])
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `output`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        return InferenceData(
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
            }
        )


def _process_configuration(comments):
    """Extract sampling information."""
    num_samples = None
    num_warmup = None
    save_warmup = None
    for comment in comments:
        comment = comment.strip("#").strip()
        if comment.startswith("num_samples"):
            num_samples = int(comment.strip("num_samples = ").strip("(Default)"))
        elif comment.startswith("num_warmup"):
            num_warmup = int(comment.strip("num_warmup = ").strip("(Default)"))
        elif comment.startswith("save_warmup"):
            save_warmup = bool(int(comment.strip("save_warmup = ").strip("(Default)")))
        elif comment.startswith("thin"):
            thin = int(comment.strip("thin = ").strip("(Default)"))

    return {
        "num_samples": num_samples,
        "num_warmup": num_warmup,
        "save_warmup": save_warmup,
        "thin": thin,
    }


def _read_output(path):
    """Read CmdStan output.csv.

    Parameters
    ----------
    path : str

    Returns
    -------
    List[DataFrame, DataFrame, List[str], List[str], List[str]]
        pandas.DataFrame
            Sample data
        pandas.DataFrame
            Sample stats
        List[str]
            Configuration information
        List[str]
            Adaptation information
        List[str]
            Timing info
    """
    chains = []
    configuration_info = []
    adaptation_info = []
    timing_info = []
    i = 0
    # Read (first) configuration and adaption
    with open(path, "r") as f_obj:
        column_names = False
        for i, line in enumerate(f_obj):
            line = line.strip()
            if line.startswith("#"):
                if column_names:
                    adaptation_info.append(line.strip())
                else:
                    configuration_info.append(line.strip())
            elif not column_names:
                column_names = True
                pconf = _process_configuration(configuration_info)
                if pconf["save_warmup"]:
                    warmup_range = range(pconf["num_warmup"] // pconf["thin"])
                    for _, _ in zip(warmup_range, f_obj):
                        continue
            else:
                break

    # Read data
    with open(path, "r") as f_obj:
        df = pd.read_csv(f_obj, comment="#")

    # split dataframe if header found multiple times
    if df.iloc[:, 0].dtype.kind == "O":
        first_col = df.columns[0]
        col_locations = first_col == df.loc[:, first_col]
        col_locations = list(col_locations.loc[col_locations].index)
        dfs = []
        for idx, last_idx in zip(col_locations, [-1] + list(col_locations[:-1])):
            df_ = deepcopy(df.loc[last_idx + 1 : idx - 1, :])
            for col in df_.columns:
                df_.loc[:, col] = pd.to_numeric(df_.loc[:, col])
            if len(df_):
                dfs.append(df_.reset_index(drop=True))
            df = df.loc[idx + 1 :, :]
        for col in df.columns:
            df.loc[:, col] = pd.to_numeric(df.loc[:, col])
        dfs.append(df)
    else:
        dfs = [df]

    for j, df in enumerate(dfs):
        if j == 0:
            # Read timing info (first) from the end of the file
            line_num = i + df.shape[0] + 1
            for k in range(5):
                line = linecache.getline(path, line_num + k).strip()
                if len(line):
                    timing_info.append(line)
            configuration_info_len = len(configuration_info)
            adaptation_info_len = len(adaptation_info)
            timing_info_len = len(timing_info)
            num_of_samples = df.shape[0]
            header_count = 1
            last_line_num = (
                configuration_info_len
                + adaptation_info_len
                + timing_info_len
                + num_of_samples
                + header_count
            )
        else:
            # header location found in the dataframe (not first)
            configuration_info = []
            adaptation_info = []
            timing_info = []

            # line number for the next dataframe in csv
            line_num = last_line_num + 1

            # row ranges
            config_start = line_num
            config_end = config_start + configuration_info_len

            # read configuration_info
            for reading_line in range(config_start, config_end):
                line = linecache.getline(path, reading_line)
                if line.startswith("#"):
                    configuration_info.append(line)
                else:
                    msg = (
                        "Invalid input file. "
                        "Header information missing from combined csv. "
                        "Configuration: {}".format(path)
                    )
                    raise ValueError(msg)

            pconf = _process_configuration(configuration_info)
            warmup_rows = pconf["save_warmup"] * pconf["num_warmup"] // pconf["thin"]
            adaption_start = config_end + 1 + warmup_rows
            adaption_end = adaption_start + adaptation_info_len
            # read adaptation_info
            for reading_line in range(adaption_start, adaption_end):
                line = linecache.getline(path, reading_line)
                if line.startswith("#"):
                    adaptation_info.append(line)
                else:
                    msg = (
                        "Invalid input file. "
                        "Header information missing from combined csv. "
                        "Adaptation: {}".format(path)
                    )
                    raise ValueError(msg)

            timing_start = adaption_end + len(df) - warmup_rows
            timing_end = timing_start + timing_info_len
            # read timing_info
            raise_timing_error = False
            for reading_line in range(timing_start, timing_end):
                line = linecache.getline(path, reading_line)
                if line.startswith("#"):
                    timing_info.append(line)
                else:
                    raise_timing_error = True
                    break
            no_elapsed_time = not any("elapsed time" in row.lower() for row in timing_info)
            if raise_timing_error or no_elapsed_time:
                msg = (
                    "Invalid input file. "
                    "Header information missing from combined csv. "
                    "Timing: {}".format(path)
                )
                raise ValueError(msg)

            last_line_num = reading_line

        # Remove warmup
        if pconf["save_warmup"]:
            saved_samples = pconf["num_samples"] // pconf["thin"]
            df = df.iloc[-saved_samples:, :]

        # Split data to sample_stats and sample
        sample_stats_columns = [col for col in df.columns if col.endswith("__")]
        sample_columns = [col for col in df.columns if col not in sample_stats_columns]

        sample_stats = df.loc[:, sample_stats_columns]
        sample_df = df.loc[:, sample_columns]

        chains.append((sample_df, sample_stats, configuration_info, adaptation_info, timing_info))

    return chains


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
    """Read Rdump output and transform to Python dictionary.

    Parameters
    ----------
    path : str

    Returns
    -------
    Dict
        key, values pairs from Rdump formatted data.
    """
    data = {}
    with open(path, "r") as f_obj:
        var = ""
        for line in f_obj:
            if "<-" in line:
                if len(var):
                    key, var = _process_data_var(var)
                    data[key] = var
                var = ""
            var += " " + line.strip()
        if len(var):
            key, var = _process_data_var(var)
            data[key] = var
    return data


def _unpack_dataframes(dfs):
    """Transform a list of pandas.DataFrames to dictionary containing ndarrays.

    Parameters
    ----------
    dfs : List[pandas.DataFrame]

    Returns
    -------
    Dict
        key, values pairs. Values are formatted to shape = (nchain, ndraws, *shape)
    """
    col_groups = defaultdict(list)
    columns = dfs[0].columns
    for col in columns:
        key, *loc = col.split(".")
        loc = tuple(int(i) - 1 for i in loc)
        col_groups[key].append((col, loc))

    chains = len(dfs)
    draws = len(dfs[0])
    sample = {}
    for key, cols_locs in col_groups.items():
        ndim = np.array([loc for _, loc in cols_locs]).max(0) + 1
        dtype = dfs[0][cols_locs[0][0]].dtype
        sample[key] = utils.full((chains, draws, *ndim), 0, dtype=dtype)
        for col, loc in cols_locs:
            for chain_id, df in enumerate(dfs):
                draw = df[col].values
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
    coords: Optional[CoordSpec] = None,
    dims: Optional[DimSpec] = None
) -> InferenceData:
    """Convert CmdStan data into an InferenceData object.

    For a usage example read the
    :doc:`Cookbook section on from_cmdstan </notebooks/InferenceDataCookbook>`

    Parameters
    ----------
    posterior : str or list of str, optional
        List of paths to output.csv files.
        CSV file can be stacked csv containing all the chains

            cat output*.csv > combined_output.csv

    posterior_predictive : str or list of str, optional
        Posterior predictive samples for the fit. If endswith ".csv" assumes file.
    predictions : str or list of str, optional
        Out of sample predictions samples for the fit. If endswith ".csv" assumes file.
    prior : str or list of str, optional
        List of paths to output.csv files
        CSV file can be stacked csv containing all the chains.

            cat output*.csv > combined_output.csv

    prior_predictive : str or list of str, optional
        Prior predictive samples for the fit. If endswith ".csv" assumes file.
    observed_data : str, optional
        Observed data used in the sampling. Path to data file in Rdump format.
    observed_data_var : str or list of str, optional
        Variable(s) used for slicing observed_data. If not defined, all
        data variables are imported.
    constant_data : str, optional
        Constant data used in the sampling. Path to data file in Rdump format.
    constant_data_var : str or list of str, optional
        Variable(s) used for slicing constant_data. If not defined, all
        data variables are imported.
    predictions_constant_data : str, optional
        Constant data for predictions used in the sampling.
        Path to data file in Rdump format.
    predictions_constant_data_var : str or list of str, optional
        Variable(s) used for slicing predictions_constant_data.
        If not defined, all data variables are imported.
    log_likelihood : str or list of str, optional
        Pointwise log_likelihood for the data.
    coords : dict of {str: array_like}, optional
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict of {str: list of str, optional
        A mapping from variables to a list of coordinate names for the variable.

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
        coords=coords,
        dims=dims,
    ).to_inference_data()
