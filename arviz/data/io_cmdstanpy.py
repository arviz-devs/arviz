"""CmdStanPy-specific conversion code."""
from collections import defaultdict
from copy import deepcopy
import logging
import re

import numpy as np
import xarray as xr

from .. import utils
from .inference_data import InferenceData
from .base import requires, dict_to_dataset, generate_dims_coords, make_attrs

_log = logging.getLogger(__name__)


class CmdStanPyConverter:
    """Encapsulate CmdStanPy specific logic."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        *,
        posterior=None,
        posterior_predictive=None,
        prior=None,
        prior_predictive=None,
        observed_data=None,
        log_likelihood=None,
        coords=None,
        dims=None
    ):
        self.posterior = posterior
        self.posterior_predictive = posterior_predictive
        self.prior = prior
        self.prior_predictive = prior_predictive
        self.observed_data = observed_data
        if isinstance(log_likelihood, (list, tuple)):
            if len(log_likelihood) == 1:
                log_likelihood = log_likelihood[0]
        self.log_likelihood = log_likelihood
        self.coords = coords
        self.dims = dims

        import cmdstanpy  # pylint: disable=import-error

        self.cmdstanpy = cmdstanpy

    @requires("posterior")
    def posterior_to_xarray(self):
        """Extract posterior samples from output csv."""
        columns = self.posterior.column_names

        # filter posterior_predictive and log_likelihood
        posterior_predictive = self.posterior_predictive
        if posterior_predictive is None:
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

        log_likelihood = self.log_likelihood
        if log_likelihood is None:
            log_likelihood = []
        else:
            log_likelihood = [col for col in columns if log_likelihood == col.split(".")[0]]

        invalid_cols = (
            posterior_predictive + log_likelihood + [col for col in columns if col.endswith("__")]
        )
        valid_cols = [col for col in columns if col not in invalid_cols]
        data = _unpack_frame(self.posterior.sample, columns, valid_cols)
        return dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims)

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from fit."""
        dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64}

        columns = self.posterior.column_names
        valid_cols = [col for col in columns if col.endswith("__")]
        # copy dims and coords
        dims = deepcopy(self.dims) if self.dims is not None else {}
        coords = deepcopy(self.coords) if self.coords is not None else {}

        log_likelihood = self.log_likelihood
        if isinstance(log_likelihood, str):
            valid_cols.append(log_likelihood)

            log_likelihood_cols = [col for col in columns if log_likelihood == col.split(".")[0]]
            # change dims and coords for log_likelihood if defined
            if log_likelihood in dims:
                dims["log_likelihood"] = dims.pop(log_likelihood)

            log_likelihood_dims = np.array(
                [list(map(int, col.split(".")[1:])) for col in log_likelihood_cols]
            )
            max_dims = log_likelihood_dims.max(0)
            max_dims = max_dims if hasattr(max_dims, "__iter__") else (max_dims,)
            default_dim_names, _ = generate_dims_coords(shape=max_dims, var_name=log_likelihood)
            log_likelihood_dim_names, _ = generate_dims_coords(
                shape=max_dims, var_name="log_likelihood"
            )
            for default_dim_name, log_likelihood_dim_name in zip(
                default_dim_names, log_likelihood_dim_names
            ):
                if default_dim_name in coords:
                    coords[log_likelihood_dim_name] = coords.pop(default_dim_name)

        data = _unpack_frame(self.posterior.sample, columns, valid_cols)
        if log_likelihood in data:
            data["log_likelihood"] = data.pop(log_likelihood)
        for s_param in list(data.keys()):
            s_param_, *_ = s_param.split(".")
            name = re.sub("__$", "", s_param_)
            name = "diverging" if name == "divergent" else name
            data[name] = data.pop(s_param).astype(dtypes.get(s_param, float))
        return dict_to_dataset(data, library=self.cmdstanpy, coords=coords, dims=dims)

    @requires("posterior")
    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        posterior_predictive = self.posterior_predictive
        columns = self.posterior.column_names

        if isinstance(posterior_predictive, str):
            posterior_predictive = [posterior_predictive]
        valid_cols = [col for col in columns if col.split(".")[0] in set(posterior_predictive)]
        data = _unpack_frame(self.posterior.sample, columns, valid_cols)
        return dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims)

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
            prior_predictive = [col for col in columns if prior_predictive == col.split(".")[0]]
        else:
            prior_predictive = [
                col for col in columns if col.split(".")[0] in set(prior_predictive)
            ]

        valid_cols = [col for col in columns if col not in set(prior_predictive)]
        data = _unpack_frame(self.posterior.sample, columns, valid_cols)
        return dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims)

    @requires("prior")
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats from fit."""
        dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64}

        columns = self.prior.column_names
        valid_cols = [col for col in columns if col.endswith("__")]
        # copy dims and coords
        dims = deepcopy(self.dims) if self.dims is not None else {}
        coords = deepcopy(self.coords) if self.coords is not None else {}

        data = _unpack_frame(self.prior.sample, columns, valid_cols)
        for s_param in list(data.keys()):
            s_param_, *_ = s_param.split(".")
            name = re.sub("__$", "", s_param_)
            name = "diverging" if name == "divergent" else name
            data[name] = data.pop(s_param).astype(dtypes.get(s_param, float))
        return dict_to_dataset(data, library=self.cmdstanpy, coords=coords, dims=dims)

    @requires("prior")
    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        prior_predictive = self.prior_predictive
        columns = self.prior.column_names

        if isinstance(prior_predictive, str):
            prior_predictive = [prior_predictive]
        valid_cols = [col for col in columns if col.split(".")[0] in set(prior_predictive)]
        data = _unpack_frame(self.prior.sample, columns, valid_cols)
        return dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims)

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
                "posterior_predictive": self.posterior_predictive_to_xarray(),
                "prior": self.prior_to_xarray(),
                "sample_stats_prior": self.sample_stats_prior_to_xarray(),
                "prior_predictive": self.prior_predictive_to_xarray(),
                "observed_data": self.observed_data_to_xarray(),
            }
        )


def _unpack_frame(data, columns, valid_cols):
    """Transform a list of pandas.DataFrames to dictionary containing ndarrays.

    Parameters
    ----------
    data : np.ndarray
    columns : list
    valid_cols : list

    Returns
    -------
    dict
        key, values pairs. Values are formatted to shape = (chains, draws, *shape)
    """
    draws, chains, *_ = data.shape

    column_groups = defaultdict(list)
    column_locs = defaultdict(list)
    # iterate flat column names
    for i, col in enumerate(columns):
        # parse parameter names e.g. X.1.2 --> X, (1,2)
        col_base, *col_tail = col.split(".")
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
    valid_base_cols = []
    # get list of parameters for extraction (basename) X.1.2 --> X
    for col in valid_cols:
        base_col, *_ = col.split(".")
        if base_col not in valid_base_cols:
            valid_base_cols.append(base_col)

    # extract each wanted parameter to ndarray with correct shape
    for key in valid_base_cols:
        ndim = dims.get(key, None)
        shape_location = column_groups.get(key, None)
        if ndim is not None:
            sample[key] = np.full((chains, draws, *ndim), np.nan)
        if shape_location is None:
            # reorder draw, chain -> chain, draw
            (i,) = column_locs[key]
            sample[key] = np.swapaxes(data[..., i], 0, 1)
        else:
            for i, shape_loc in zip(column_locs[key], shape_location):
                # location to insert extracted array
                shape_loc = tuple([Ellipsis] + [j - 1 for j in shape_loc])
                # reorder draw, chain -> chain, draw and insert to ndarray
                sample[key][shape_loc] = np.swapaxes(data[..., i], 0, 1)
    return sample


def from_cmdstanpy(
    posterior=None,
    *,
    posterior_predictive=None,
    prior=None,
    prior_predictive=None,
    observed_data=None,
    log_likelihood=None,
    coords=None,
    dims=None
):
    """Convert CmdStanPy data into an InferenceData object.

    Parameters
    ----------
    posterior : CmdStanMCMC object
        CmdStanPy CmdStanMCMC
    posterior_predictive : str, list of str
        Posterior predictive samples for the fit.
    prior : CmdStanMCMC
        CmdStanPy CmdStanMCMC
    prior_predictive : str, list of str
        Prior predictive samples for the fit.
    observed_data : dict
        Observed data used in the sampling.
    log_likelihood : str
        Pointwise log_likelihood for the data.
    coords : dict of str or dict of iterable
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict of str or list of str
        A mapping from variables to a list of coordinate names for the variable.

    Returns
    -------
    InferenceData object
    """
    return CmdStanPyConverter(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        prior=prior,
        prior_predictive=prior_predictive,
        observed_data=observed_data,
        log_likelihood=log_likelihood,
        coords=coords,
        dims=dims,
    ).to_inference_data()
