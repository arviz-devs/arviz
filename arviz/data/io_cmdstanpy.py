"""CmdStanPy-specific conversion code."""
import logging
import re
from collections import defaultdict
from copy import deepcopy

import numpy as np
import xarray as xr

from .. import utils
from ..rcparams import rcParams
from .base import dict_to_dataset, generate_dims_coords, make_attrs, requires
from .inference_data import InferenceData

_log = logging.getLogger(__name__)


class CmdStanPyConverter:
    """Encapsulate CmdStanPy specific logic."""

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
        constant_data=None,
        predictions_constant_data=None,
        log_likelihood=None,
        coords=None,
        dims=None,
        save_warmup=None,
    ):
        self.posterior = posterior
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions
        self.prior = prior
        self.prior_predictive = prior_predictive
        self.observed_data = observed_data
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.log_likelihood = log_likelihood
        self.coords = coords
        self.dims = dims

        self.save_warmup = rcParams["data.save_warmup"] if save_warmup is None else save_warmup

        import cmdstanpy  # pylint: disable=import-error

        self.cmdstanpy = cmdstanpy

    @requires("posterior")
    def posterior_to_xarray(self):
        """Extract posterior samples from output csv."""
        columns = self.posterior.column_names

        # filter posterior_predictive, predictions and log_likelihood
        posterior_predictive = self.posterior_predictive
        if posterior_predictive is None:
            posterior_predictive = []
        elif isinstance(posterior_predictive, str):
            posterior_predictive = [
                col for col in columns if posterior_predictive == col.split("[")[0].split(".")[0]
            ]
        else:
            posterior_predictive = [
                col
                for col in columns
                if any(item == col.split("[")[0].split(".")[0] for item in posterior_predictive)
            ]

        predictions = self.predictions
        if predictions is None:
            predictions = []
        elif isinstance(predictions, str):
            predictions = [col for col in columns if predictions == col.split("[")[0].split(".")[0]]
        else:
            predictions = [
                col
                for col in columns
                if any(item == col.split("[")[0].split(".")[0] for item in predictions)
            ]

        log_likelihood = self.log_likelihood
        if log_likelihood is None:
            log_likelihood = []
        elif isinstance(log_likelihood, str):
            log_likelihood = [
                col for col in columns if log_likelihood == col.split("[")[0].split(".")[0]
            ]
        else:
            log_likelihood = [
                col
                for col in columns
                if any(item == col.split("[")[0].split(".")[0] for item in log_likelihood)
            ]

        invalid_cols = set(
            posterior_predictive
            + predictions
            + log_likelihood
            + [col for col in columns if col.endswith("__")]
        )
        valid_cols = [col for col in columns if col not in invalid_cols]
        data, data_warmup = _unpack_frame(
            self.posterior,
            columns,
            valid_cols,
            self.save_warmup,
        )

        return (
            dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims),
            dict_to_dataset(
                data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims
            ),
        )

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from fit."""
        dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64}

        columns = self.posterior.column_names
        valid_cols = [col for col in columns if col.endswith("__")]
        data, data_warmup = _unpack_frame(
            self.posterior,
            columns,
            valid_cols,
            self.save_warmup,
        )

        for s_param in list(data.keys()):
            s_param_, *_ = s_param.split(".")
            name = re.sub("__$", "", s_param_)
            name = "diverging" if name == "divergent" else name
            data[name] = data.pop(s_param).astype(dtypes.get(s_param, float))
            if data_warmup:
                data_warmup[name] = data_warmup.pop(s_param).astype(dtypes.get(s_param, float))
        return (
            dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims),
            dict_to_dataset(
                data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims
            ),
        )

    @requires("posterior")
    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        posterior_predictive = self.posterior_predictive
        columns = self.posterior.column_names

        if isinstance(posterior_predictive, str):
            posterior_predictive = [posterior_predictive]
        posterior_predictive = set(posterior_predictive)
        valid_cols = [
            col for col in columns if col.split("[")[0].split(".")[0] in posterior_predictive
        ]
        data, data_warmup = _unpack_frame(
            self.posterior,
            columns,
            valid_cols,
            self.save_warmup,
        )

        return (
            dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims),
            dict_to_dataset(
                data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims
            ),
        )

    @requires("posterior")
    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert out of sample predictions samples to xarray."""
        predictions = self.predictions
        columns = self.posterior.column_names

        if isinstance(predictions, str):
            predictions = [predictions]
        predictions = set(predictions)
        valid_cols = [col for col in columns if col.split("[")[0].split(".")[0] in set(predictions)]
        data, data_warmup = _unpack_frame(
            self.posterior,
            columns,
            valid_cols,
            self.save_warmup,
        )

        return (
            dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims),
            dict_to_dataset(
                data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims
            ),
        )

    @requires("posterior")
    @requires("log_likelihood")
    def log_likelihood_to_xarray(self):
        """Convert elementwise log likelihood samples to xarray."""
        log_likelihood = self.log_likelihood
        columns = self.posterior.column_names

        if isinstance(log_likelihood, str):
            log_likelihood = [log_likelihood]
        log_likelihood = set(log_likelihood)
        valid_cols = [col for col in columns if col.split("[")[0].split(".")[0] in log_likelihood]
        data, data_warmup = _unpack_frame(
            self.posterior,
            columns,
            valid_cols,
            self.save_warmup,
        )

        return (
            dict_to_dataset(
                data,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                skip_event_dims=True,
            ),
            dict_to_dataset(
                data_warmup,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                skip_event_dims=True,
            ),
        )

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        # filter prior_predictive
        columns = self.prior.column_names

        # filter posterior_predictive and log_likelihood
        prior_predictive = self.prior_predictive
        if prior_predictive is None:
            prior_predictive = []
        elif isinstance(prior_predictive, str):
            prior_predictive = [
                col for col in columns if prior_predictive == col.split("[")[0].split(".")[0]
            ]
        else:
            prior_predictive = [
                col for col in columns if col.split("[")[0].split(".")[0] in set(prior_predictive)
            ]

        invalid_cols = set(prior_predictive + [col for col in columns if col.endswith("__")])

        valid_cols = [col for col in columns if col not in invalid_cols]

        data, data_warmup = _unpack_frame(
            self.prior,
            columns,
            valid_cols,
            self.save_warmup,
        )

        return (
            dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims),
            dict_to_dataset(
                data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims
            ),
        )

    @requires("prior")
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats from fit."""
        dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64}

        columns = self.prior.column_names
        valid_cols = [col for col in columns if col.endswith("__")]
        # copy dims and coords
        dims = deepcopy(self.dims) if self.dims is not None else {}
        coords = deepcopy(self.coords) if self.coords is not None else {}

        data, data_warmup = _unpack_frame(
            self.prior,
            columns,
            valid_cols,
            self.save_warmup,
        )

        for s_param in list(data.keys()):
            s_param_, *_ = s_param.split(".")
            name = re.sub("__$", "", s_param_)
            name = "diverging" if name == "divergent" else name
            data[name] = data.pop(s_param).astype(dtypes.get(s_param, float))
            if data_warmup:
                data_warmup[name] = data_warmup.pop(s_param).astype(dtypes.get(s_param, float))
        return (
            dict_to_dataset(data, library=self.cmdstanpy, coords=coords, dims=dims),
            dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=coords, dims=dims),
        )

    @requires("prior")
    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        prior_predictive = self.prior_predictive
        columns = self.prior.column_names

        if isinstance(prior_predictive, str):
            prior_predictive = [prior_predictive]
        prior_predictive = set(prior_predictive)
        valid_cols = [col for col in columns if col.split("[")[0].split(".")[0] in prior_predictive]
        data, data_warmup = _unpack_frame(
            self.prior,
            columns,
            valid_cols,
            self.save_warmup,
        )

        return (
            dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims),
            dict_to_dataset(
                data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims
            ),
        )

    @requires("observed_data")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        observed_data = {}
        for key, vals in self.observed_data.items():
            vals = utils.one_de(vals)
            val_dims = self.dims.get(key) if self.dims is not None else None
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            observed_data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=observed_data, attrs=make_attrs(library=self.cmdstanpy))

    @requires("constant_data")
    def constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        constant_data = {}
        for key, vals in self.constant_data.items():
            vals = utils.one_de(vals)
            val_dims = self.dims.get(key) if self.dims is not None else None
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            constant_data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=constant_data, attrs=make_attrs(library=self.cmdstanpy))

    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        predictions_constant_data = {}
        for key, vals in self.predictions_constant_data.items():
            vals = utils.one_de(vals)
            val_dims = self.dims.get(key) if self.dims is not None else None
            val_dims, coords = generate_dims_coords(
                vals.shape, key, dims=val_dims, coords=self.coords
            )
            predictions_constant_data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(
            data_vars=predictions_constant_data, attrs=make_attrs(library=self.cmdstanpy)
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
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "predictions": self.predictions_to_xarray(),
                "prior": self.prior_to_xarray(),
                "sample_stats_prior": self.sample_stats_prior_to_xarray(),
                "prior_predictive": self.prior_predictive_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
                "constant_data": self.constant_data_to_xarray(),
                "predictions_constant_data": self.predictions_constant_data_to_xarray(),
                "log_likelihood": self.log_likelihood_to_xarray(),
            },
        )


def _unpack_frame(fit, columns, valid_cols, save_warmup):
    """Transform fit to dictionary containing ndarrays.

    Parameters
    ----------
    data: cmdstanpy.CmdStanMCMC
    columns: list
    valid_cols: list
    save_warmup: bool

    Returns
    -------
    dict
        key, values pairs. Values are formatted to shape = (chains, draws, *shape)
    """
    if save_warmup and not fit._save_warmup:  # pylint: disable=protected-access
        save_warmup = False
    if hasattr(fit, "draws"):
        data = fit.draws(inc_warmup=save_warmup)
        if save_warmup:
            num_warmup = fit._draws_warmup  # pylint: disable=protected-access
            data_warmup = data[:num_warmup]
            data = data[num_warmup:]
    else:
        data = fit.sample
        if save_warmup:
            data_warmup = fit.warmup[: data.shape[0]]

    draws, chains, *_ = data.shape
    if save_warmup:
        draws_warmup, *_ = data_warmup.shape

    column_groups = defaultdict(list)
    column_locs = defaultdict(list)
    # iterate flat column names
    for i, col in enumerate(columns):
        if "." in col:
            # parse parameter names e.g. X.1.2 --> X, (1,2)
            col_base, *col_tail = col.split(".")
        else:
            # parse parameter names e.g. X[1,2] --> X, (1,2)
            col_base, *col_tail = col.replace("]", "").replace("[", ",").split(",")
        if len(col_tail):
            # gather nD array locations
            column_groups[col_base].append(tuple(map(int, col_tail)))
        # gather raw data locations for each parameter
        column_locs[col_base].append(i)
    dims = {}
    for colname, col_dims in column_groups.items():
        # gather parameter dimensions (assumes dense arrays)
        dims[colname] = tuple(np.array(col_dims).max(0))
    sample = {}
    sample_warmup = {}
    valid_base_cols = []
    # get list of parameters for extraction (basename) X.1.2 --> X
    for col in valid_cols:
        base_col = col.split("[")[0].split(".")[0]
        if base_col not in valid_base_cols:
            valid_base_cols.append(base_col)

    # extract each wanted parameter to ndarray with correct shape
    for key in valid_base_cols:
        ndim = dims.get(key, None)
        shape_location = column_groups.get(key, None)
        if ndim is not None:
            sample[key] = np.full((chains, draws, *ndim), np.nan)
            if save_warmup:
                sample_warmup[key] = np.full((chains, draws_warmup, *ndim), np.nan)
        if shape_location is None:
            # reorder draw, chain -> chain, draw
            (i,) = column_locs[key]
            sample[key] = np.swapaxes(data[..., i], 0, 1)
            if save_warmup:
                sample_warmup[key] = np.swapaxes(data_warmup[..., i], 0, 1)
        else:
            for i, shape_loc in zip(column_locs[key], shape_location):
                # location to insert extracted array
                shape_loc = tuple([Ellipsis] + [j - 1 for j in shape_loc])
                # reorder draw, chain -> chain, draw and insert to ndarray
                sample[key][shape_loc] = np.swapaxes(data[..., i], 0, 1)
                if save_warmup:
                    sample_warmup[key][shape_loc] = np.swapaxes(data_warmup[..., i], 0, 1)
    return sample, sample_warmup


def from_cmdstanpy(
    posterior=None,
    *,
    posterior_predictive=None,
    predictions=None,
    prior=None,
    prior_predictive=None,
    observed_data=None,
    constant_data=None,
    predictions_constant_data=None,
    log_likelihood=None,
    coords=None,
    dims=None,
    save_warmup=None,
):
    """Convert CmdStanPy data into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_cmdstanpy <creating_InferenceData>`

    Parameters
    ----------
    posterior : CmdStanMCMC object
        CmdStanPy CmdStanMCMC
    posterior_predictive : str, list of str
        Posterior predictive samples for the fit.
    predictions : str, list of str
        Out of sample prediction samples for the fit.
    prior : CmdStanMCMC
        CmdStanPy CmdStanMCMC
    prior_predictive : str, list of str
        Prior predictive samples for the fit.
    observed_data : dict
        Observed data used in the sampling.
    constant_data : dict
        Constant data used in the sampling.
    predictions_constant_data : dict
        Constant data for predictions used in the sampling.
    log_likelihood : str, list of str
        Pointwise log_likelihood for the data.
    coords : dict of str or dict of iterable
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict of str or list of str
        A mapping from variables to a list of coordinate names for the variable.
    save_warmup : bool
        Save warmup iterations into InferenceData object, if found in the input files.
        If not defined, use default defined by the rcParams.

    Returns
    -------
    InferenceData object
    """
    return CmdStanPyConverter(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        predictions=predictions,
        prior=prior,
        prior_predictive=prior_predictive,
        observed_data=observed_data,
        constant_data=constant_data,
        predictions_constant_data=predictions_constant_data,
        log_likelihood=log_likelihood,
        coords=coords,
        dims=dims,
        save_warmup=save_warmup,
    ).to_inference_data()
