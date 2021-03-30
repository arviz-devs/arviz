"""CmdStanPy-specific conversion code."""
import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from ..rcparams import rcParams
from .base import dict_to_dataset, infer_stan_dtypes, make_attrs, requires
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
        index_origin=None,
        coords=None,
        dims=None,
        save_warmup=None,
        dtypes=None,
    ):
        self.posterior = posterior  # CmdStanPy CmdStanMCMC object
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions
        self.prior = prior
        self.prior_predictive = prior_predictive
        self.observed_data = observed_data
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.log_likelihood = (
            rcParams["data.log_likelihood"] if log_likelihood is None else log_likelihood
        )
        self.index_origin = index_origin
        self.coords = coords
        self.dims = dims

        self.save_warmup = rcParams["data.save_warmup"] if save_warmup is None else save_warmup

        import cmdstanpy  # pylint: disable=import-error

        if dtypes is None:
            dtypes = {}
        elif isinstance(dtypes, cmdstanpy.model.CmdStanModel):
            model_code = dtypes.code()
            dtypes = infer_stan_dtypes(model_code)
        elif isinstance(dtypes, str):
            dtypes_path = Path(dtypes)
            if dtypes_path.exists():
                with dtypes_path.open("r", encoding="UTF-8") as f_obj:
                    model_code = f_obj.read()
            else:
                model_code = dtypes
            dtypes = infer_stan_dtypes(model_code)

        self.dtypes = dtypes

        if hasattr(self.posterior, "metadata"):
            if self.log_likelihood is True and "log_lik" in self.posterior.metadata.stan_vars_cols:
                self.log_likelihood = ["log_lik"]
        elif hasattr(self.posterior, "stan_vars_cols"):
            if self.log_likelihood is True and "log_lik" in self.posterior.stan_vars_cols:
                self.log_likelihood = ["log_lik"]
        else:
            if (
                self.log_likelihood is True
                and self.posterior is not None
                and any(name.split("[")[0] == "log_lik" for name in self.posterior.column_names)
            ):
                self.log_likelihood = ["log_lik"]

        if isinstance(self.log_likelihood, bool):
            self.log_likelihood = None

        self.cmdstanpy = cmdstanpy

    @requires("posterior")
    def posterior_to_xarray(self):
        """Extract posterior samples from output csv."""
        if not (hasattr(self.posterior, "metadata") or hasattr(self.posterior, "stan_vars_cols")):
            return self.posterior_to_xarray_pre_v_0_9_68()

        if hasattr(self.posterior, "metadata"):
            items = list(self.posterior.metadata.stan_vars_cols.keys())
        else:
            items = list(self.posterior.stan_vars_cols.keys())
        if self.posterior_predictive is not None:
            try:
                items = _filter(items, self.posterior_predictive)
            except ValueError:
                pass
        if self.predictions is not None:
            try:
                items = _filter(items, self.predictions)
            except ValueError:
                pass
        if self.log_likelihood is not None:
            try:
                items = _filter(items, self.log_likelihood)
            except ValueError:
                pass

        valid_cols = []
        for item in items:
            if hasattr(self.posterior, "metadata"):
                valid_cols.extend(self.posterior.metadata.stan_vars_cols[item])
            else:
                valid_cols.extend(self.posterior.stan_vars_cols[item])

        data, data_warmup = _unpack_fit(
            self.posterior,
            items,
            self.save_warmup,
            self.dtypes,
        )

        # copy dims and coords  - Mitzi question:  why???
        dims = deepcopy(self.dims) if self.dims is not None else {}
        coords = deepcopy(self.coords) if self.coords is not None else {}

        return (
            dict_to_dataset(
                data,
                library=self.cmdstanpy,
                coords=coords,
                dims=dims,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                library=self.cmdstanpy,
                coords=coords,
                dims=dims,
                index_origin=self.index_origin,
            ),
        )

    @requires("posterior")
    def sample_stats_to_xarray(self):
        """Extract sample_stats from prosterior fit."""
        return self.stats_to_xarray(self.posterior)

    @requires("prior")
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats from prior fit."""
        return self.stats_to_xarray(self.prior)

    def stats_to_xarray(self, fit):
        """Extract sample_stats from fit."""
        if not (hasattr(fit, "metadata") or hasattr(fit, "sampler_vars_cols")):
            return self.sample_stats_to_xarray_pre_v_0_9_68(fit)

        dtypes = {
            "divergent__": bool,
            "n_leapfrog__": np.int64,
            "treedepth__": np.int64,
            **self.dtypes,
        }
        if hasattr(fit, "metadata"):
            items = list(fit.metadata._method_vars_cols.keys())  # pylint: disable=protected-access
        else:
            items = list(fit.sampler_vars_cols.keys())
        rename_dict = {
            "divergent": "diverging",
            "n_leapfrog": "n_steps",
            "treedepth": "tree_depth",
            "stepsize": "step_size",
            "accept_stat": "acceptance_rate",
        }

        data, data_warmup = _unpack_fit(
            fit,
            items,
            self.save_warmup,
            self.dtypes,
        )
        for item in items:
            name = re.sub("__$", "", item)
            name = rename_dict.get(name, name)
            data[name] = data.pop(item).astype(dtypes.get(item, float))
            if data_warmup:
                data_warmup[name] = data_warmup.pop(item).astype(dtypes.get(item, float))
        return (
            dict_to_dataset(
                data,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            ),
        )

    @requires("posterior")
    @requires("posterior_predictive")
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        return self.predictive_to_xarray(self.posterior_predictive, self.posterior)

    @requires("prior")
    @requires("prior_predictive")
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        return self.predictive_to_xarray(self.prior_predictive, self.prior)

    def predictive_to_xarray(self, names, fit):
        """Convert predictive samples to xarray."""
        predictive = _as_set(names)

        if hasattr(fit, "metadata") or hasattr(fit, "stan_vars_cols"):
            data, data_warmup = _unpack_fit(
                fit,
                predictive,
                self.save_warmup,
                self.dtypes,
            )
        else:  # pre_v_0_9_68
            valid_cols = _filter_columns(fit.column_names, predictive)
            data, data_warmup = _unpack_frame(
                fit,
                fit.column_names,
                valid_cols,
                self.save_warmup,
                self.dtypes,
            )

        return (
            dict_to_dataset(
                data,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            ),
        )

    @requires("posterior")
    @requires("predictions")
    def predictions_to_xarray(self):
        """Convert out of sample predictions samples to xarray."""
        predictions = _as_set(self.predictions)

        if hasattr(self.posterior, "metadata") or hasattr(self.posterior, "stan_vars_cols"):
            data, data_warmup = _unpack_fit(
                self.posterior,
                predictions,
                self.save_warmup,
                self.dtypes,
            )
        else:  # pre_v_0_9_68
            columns = self.posterior.column_names
            valid_cols = _filter_columns(columns, predictions)
            data, data_warmup = _unpack_frame(
                self.posterior,
                columns,
                valid_cols,
                self.save_warmup,
                self.dtypes,
            )

        return (
            dict_to_dataset(
                data,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            ),
        )

    @requires("posterior")
    @requires("log_likelihood")
    def log_likelihood_to_xarray(self):
        """Convert elementwise log likelihood samples to xarray."""
        log_likelihood = _as_set(self.log_likelihood)

        if hasattr(self.posterior, "metadata") or hasattr(self.posterior, "stan_vars_cols"):
            data, data_warmup = _unpack_fit(
                self.posterior,
                log_likelihood,
                self.save_warmup,
                self.dtypes,
            )
        else:  # pre_v_0_9_68
            columns = self.posterior.column_names
            valid_cols = _filter_columns(columns, log_likelihood)
            data, data_warmup = _unpack_frame(
                self.posterior,
                columns,
                valid_cols,
                self.save_warmup,
                self.dtypes,
            )
        if isinstance(self.log_likelihood, dict):
            data = {obs_name: data[lik_name] for obs_name, lik_name in self.log_likelihood.items()}
            if data_warmup:
                data_warmup = {
                    obs_name: data_warmup[lik_name]
                    for obs_name, lik_name in self.log_likelihood.items()
                }
        return (
            dict_to_dataset(
                data,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
                skip_event_dims=True,
            ),
            dict_to_dataset(
                data_warmup,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
                skip_event_dims=True,
            ),
        )

    @requires("prior")
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        if hasattr(self.posterior, "metadata") or hasattr(self.prior, "stan_vars_cols"):
            if hasattr(self.posterior, "metadata"):
                items = list(self.prior.metadata.stan_vars_cols.keys())
            else:
                items = list(self.prior.stan_vars_cols.keys())
            if self.prior_predictive is not None:
                try:
                    items = _filter(items, self.prior_predictive)
                except ValueError:
                    pass
            data, data_warmup = _unpack_fit(
                self.prior,
                items,
                self.save_warmup,
                self.dtypes,
            )
        else:  # pre_v_0_9_68
            columns = self.prior.column_names
            prior_predictive = _as_set(self.prior_predictive)
            prior_predictive = _filter_columns(columns, prior_predictive)

            invalid_cols = set(prior_predictive + [col for col in columns if col.endswith("__")])
            valid_cols = [col for col in columns if col not in invalid_cols]

            data, data_warmup = _unpack_frame(
                self.prior,
                columns,
                valid_cols,
                self.save_warmup,
                self.dtypes,
            )

        return (
            dict_to_dataset(
                data,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            ),
        )

    @requires("observed_data")
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        return dict_to_dataset(
            self.observed_data,
            library=self.cmdstanpy,
            coords=self.coords,
            dims=self.dims,
            default_dims=[],
            index_origin=self.index_origin,
        )

    @requires("constant_data")
    def constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        return dict_to_dataset(
            self.constant_data,
            library=self.cmdstanpy,
            coords=self.coords,
            dims=self.dims,
            default_dims=[],
            index_origin=self.index_origin,
        )

    @requires("predictions_constant_data")
    def predictions_constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        return dict_to_dataset(
            self.predictions_constant_data,
            library=self.cmdstanpy,
            coords=self.coords,
            dims=self.dims,
            attrs=make_attrs(library=self.cmdstanpy),
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

    @requires("posterior")
    def posterior_to_xarray_pre_v_0_9_68(self):
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
            self.dtypes,
        )

        return (
            dict_to_dataset(
                data,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            ),
        )

    def sample_stats_to_xarray_pre_v_0_9_68(self, fit):
        """Extract sample_stats from fit."""
        dtypes = {"divergent__": bool, "n_leapfrog__": np.int64, "treedepth__": np.int64}
        columns = fit.column_names
        valid_cols = [col for col in columns if col.endswith("__")]
        data, data_warmup = _unpack_frame(
            fit,
            columns,
            valid_cols,
            self.save_warmup,
            self.dtypes,
        )
        for s_param in list(data.keys()):
            s_param_, *_ = s_param.split(".")
            name = re.sub("__$", "", s_param_)
            name = "diverging" if name == "divergent" else name
            data[name] = data.pop(s_param).astype(dtypes.get(s_param, float))
            if data_warmup:
                data_warmup[name] = data_warmup.pop(s_param).astype(dtypes.get(s_param, float))
        return (
            dict_to_dataset(
                data,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            ),
            dict_to_dataset(
                data_warmup,
                library=self.cmdstanpy,
                coords=self.coords,
                dims=self.dims,
                index_origin=self.index_origin,
            ),
        )


def _as_set(spec):
    """Uniform representation for args which be name or list of names."""
    if spec is None:
        return []
    if isinstance(spec, str):
        return [spec]
    try:
        return set(spec.values())
    except AttributeError:
        return set(spec)


def _filter(names, spec):
    """Remove names from list of names."""
    if isinstance(spec, str):
        names.remove(spec)
    elif isinstance(spec, list):
        for item in spec:
            names.remove(item)
    elif isinstance(spec, dict):
        for item in spec.values():
            names.remove(item)
    return names


def _filter_columns(columns, spec):
    """Parse variable name from column label, removing element index, if any."""
    return [col for col in columns if col.split("[")[0].split(".")[0] in spec]


def _unpack_fit(fit, items, save_warmup, dtypes):
    """Transform fit to dictionary containing ndarrays.

    Parameters
    ----------
    data: cmdstanpy.CmdStanMCMC
    items: list
    save_warmup: bool
    dtypes: dict

    Returns
    -------
    dict
        key, values pairs. Values are formatted to shape = (chains, draws, *shape)
    """
    num_warmup = 0
    if save_warmup:
        if not fit._save_warmup:  # pylint: disable=protected-access
            save_warmup = False
        else:
            num_warmup = fit.num_draws_warmup

    nchains = fit.chains
    draws = np.swapaxes(fit.draws(inc_warmup=save_warmup), 0, 1)
    sample = {}
    sample_warmup = {}

    stan_vars_cols = fit.metadata.stan_vars_cols if hasattr(fit, "metadata") else fit.stan_vars_cols
    sampler_vars_cols = (
        fit.metadata._method_vars_cols  # pylint: disable=protected-access
        if hasattr(fit, "metadata")
        else fit.sampler_vars_cols
    )
    for item in items:
        if item in stan_vars_cols:
            col_idxs = stan_vars_cols[item]
            raw_draws = fit.stan_variable(item, inc_warmup=save_warmup)
            raw_draws = np.swapaxes(
                raw_draws.reshape((-1, nchains, *raw_draws.shape[1:]), order="F"), 0, 1
            )
        elif item in sampler_vars_cols:
            col_idxs = sampler_vars_cols[item]
            raw_draws = draws[..., col_idxs[0]]
        else:
            raise ValueError(f"fit data, unknown variable: {item}")
        raw_draws = raw_draws.astype(dtypes.get(item))
        if save_warmup:
            sample_warmup[item] = raw_draws[:, :num_warmup, ...]
            sample[item] = raw_draws[:, num_warmup:, ...]
        else:
            sample[item] = raw_draws

    return sample, sample_warmup


def _unpack_frame(fit, columns, valid_cols, save_warmup, dtypes):
    """Transform fit to dictionary containing ndarrays.

    Called when fit object created by cmdstanpy version < 0.9.68

    Parameters
    ----------
    data: cmdstanpy.CmdStanMCMC
    columns: list
    valid_cols: list
    save_warmup: bool
    dtypes: dict

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

    for key, dtype in dtypes.items():
        if key in sample:
            sample[key] = sample[key].astype(dtype)
            if save_warmup:
                if key in sample_warmup:
                    sample_warmup[key] = sample_warmup[key].astype(dtype)
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
    index_origin=None,
    coords=None,
    dims=None,
    save_warmup=None,
    dtypes=None,
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
    log_likelihood : str, list of str, dict of {str: str}, optional
        Pointwise log_likelihood for the data. If a dict, its keys should represent var_names
        from the corresponding observed data and its values the stan variable where the
        data is stored. By default, if a variable ``log_lik`` is present in the Stan model,
        it will be retrieved as pointwise log likelihood values. Use ``False``
        or set ``data.log_likelihood`` to false to avoid this behaviour.
    index_origin : int, optional
        Starting value of integer coordinate values. Defaults to the value in rcParam
        ``data.index_origin``.
    coords : dict of str or dict of iterable
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict of str or list of str
        A mapping from variables to a list of coordinate names for the variable.
    save_warmup : bool
        Save warmup iterations into InferenceData object, if found in the input files.
        If not defined, use default defined by the rcParams.
    dtypes: dict or str or cmdstanpy.CmdStanModel
        A dictionary containing dtype information (int, float) for parameters.
        If input is a string, it is assumed to be a model code or path to model code file.
        Model code can extracted from cmdstanpy.CmdStanModel object.

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
        index_origin=index_origin,
        coords=coords,
        dims=dims,
        save_warmup=save_warmup,
        dtypes=dtypes,
    ).to_inference_data()
