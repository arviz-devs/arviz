"""PyStan-specific conversion code."""
from collections import defaultdict
from copy import deepcopy
from glob import glob
import linecache
import re


import numpy as np
import pandas as pd
import xarray as xr

from .inference_data import InferenceData
from .base import requires, dict_to_dataset, generate_dims_coords


class CmdStanConverter:
    """Encapsulate CmdStan specific logic."""

    def __init__(self, *, output=None, prior=None, posterior_predictive=None,
                 observed_data=None, observed_data_var=None,
                 log_likelihood=None, coords=None, dims=None):
        self.output = glob(output) if isinstance(output, str) else output
        if isinstance(prior, str) and prior.endswith(".csv"):
            prior = glob(prior)
        self.prior = prior
        self.posterior_predictive = posterior_predictive
        self.observed_data = observed_data
        self.observed_data_var = observed_data_var
        self.log_likelihood = log_likelihood
        self.coords = coords if coords is not None else {}
        self.dims = dims if dims is not None else {}
        self.chains = len(output)
        self.sample_stats = None
        self.posterior = None
        # populate posterior and sample_Stats
        self.parse_output()

    @requires('output')
    def parse_output(self):
        chain_data = []
        for path in self.output:
            sample, sample_stats, config, adaptation, timing = read_output(path)

            chain_data.append({
                'sample' : sample,
                'sample_stats' : sample_stats,
                'configuration_info' : config,
                'adaptation_info' : adaptation,
                'timing_info' : timing,
            })
        self.posterior = [item['sample'] for item in chain_data]
        self.sample_stats = [item['sample_stats'] for item in chain_data]

    @requires('posterior')
    def posterior_to_xarray(self):
        """Extract posterior samples from output csv."""
        columns = self.posterior[0].columns

        # filter posterior_predictive and log_likelihood
        post_pred = self.posterior_predictive
        if post_pred is None or isinstance(post_pred, dict):
            post_pred = []
        elif isinstance(post_pred, str):
            post_pred = [post_pred]
        log_lik = self.log_likelihood
        if not isinstance(log_lik, str):
            log_lik = []
        else:
            log_lik = [log_lik]

        valid_cols = [col for col in columns if col not in post_pred+log_lik]
        data = unpack_dataframes([item[valid_cols] for item in self.posterior])
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('sample_stats')
    def sample_stats_to_xarray(self):
        """Extract sample_stats from fit."""
        dtypes = {
            'divergent__' : bool,
            'n_leapfrog__' : np.int64,
            'treedepth__' : np.int64,
        }

        rename_key = {
            'accept_stat__' : 'accept_stat',
            'divergent__' : 'diverging',
            'energy__' : 'energy',
            'lp__' : 'lp',
            'n_leapfrog__' : 'n_leapfrog',
            'stepsize__' : 'stepsize',
            'treedepth__' : 'treedepth',
        }
        sampler_params = self.sample_stats
        log_likelihood = self.log_likelihood
        if log_likelihood is not None:
            if isinstance(log_likelihood, str):
                if self.posterior is None:
                    log_likelihood = None
                else:
                    log_likelihood_vals = [item[log_likelihood] for item in self.posterior]
        # copy dims and coords
        dims = deepcopy(self.dims) if self.dims is not None else {}
        coords = deepcopy(self.coords) if self.coords is not None else {}

        if log_likelihood is not None:
            # Add log_likelihood to sampler_params
            for i, _ in enumerate(sampler_params):
                # slice log_likelihood to keep dimensions
                sampler_params[i]['log_likelihood'] = log_likelihood_vals[i]
            # change dims and coords for log_likelihood if defined
            if isinstance(log_likelihood, str) and log_likelihood in dims:
                dims["log_likelihood"] = dims.pop(log_likelihood)
            if isinstance(log_likelihood, str) and log_likelihood in coords:
                coords["log_likelihood"] = coords.pop(log_likelihood)
        data = {}
        for key in sampler_params[0]:
            name = rename_key.get(key, re.sub('__$', "", key))
            data[name] = np.vstack([j[key].astype(dtypes.get(key)) for j in sampler_params])
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('posterior')
    @requires('posterior_predictive')
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        var_names = self.posterior_predictive
        if isinstance(var_names, str):
            var_names = [var_names]
        data = unpack_dataframes([item[var_names] for item in self.posterior])
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('posterior')
    @requires('prior')
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        chains = []
        for path in self.prior:
            prior, *_ = read_output(path)
            chains.append(prior)
        data = unpack_dataframes(chains)
        return dict_to_dataset(data, coords=self.coords, dims=self.dims)

    @requires('posterior')
    @requires('observed_data')
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        observed_data_raw = read_data(self.observed_data)
        variables = self.observed_data_var
        if isinstance(variables, str):
            variables = [variables]
        observed_data = {}
        for key, vals in observed_data_raw.items():
            if variables is not None and key not in variables:
                continue
            vals = np.atleast_1d(vals)
            val_dims = self.dims.get(key)
            val_dims, coords = generate_dims_coords(vals.shape, key,
                                                    dims=val_dims, coords=self.coords)
            observed_data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
        return xr.Dataset(data_vars=observed_data)

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `output`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        return InferenceData(**{
            'posterior': self.posterior_to_xarray(),
            'sample_stats': self.sample_stats_to_xarray(),
            'posterior_predictive' : self.posterior_predictive_to_xarray(),
            'prior' : self.prior_to_xarray(),
            'observed_data' : self.observed_data_to_xarray(),
        })

def _process_configuration(comments):
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

    return {'num_samples'  : num_samples,
            'num_warmup' : num_warmup,
            'save_warmup' : save_warmup,
            'thin' : thin,
           }

def read_output(path):
    """Function for reading CmdStan output.csv

    Parameters
    ----------
    path : str

    Returns
    -------
    pandas.DataFrame
        Sample data
    pandas.DataFrame
        Sample stats
    list[str]
        Configuration information
    list[str]
        Adaptation information
    list[str]
        Timing info
    """
    configuration_info = []
    adaptation_info = []
    timing_info = []
    i = 0
    # Read configuration and adaption
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
            else:
                break

    # Read data
    with open(path, "r") as f_obj:
        df = pd.read_csv(f_obj, comment="#")

    # Read timing info from the end of the file
    linelen = 1
    linenum = i + df.shape[0] + 1
    while linelen:
        line = linecache.getline(path, linenum)
        linenum += 1
        linelen = len(line)
        if linelen:
            line = line.strip()
            if line and line != '#':
                timing_info.append(line)

    # Remove warmup
    processed_config = _process_configuration(configuration_info)
    if processed_config['save_warmup']:
        saved_samples = processed_config['num_samples']//processed_config['thin']
        df = df.iloc[-saved_samples:, :]

    # Split data to sample_stats and sample
    sample_stats_columns = [col for col in df.columns if col.endswith("__")]
    sample_columns = [col for col in df.columns if col not in sample_stats_columns]

    sample_stats = df.loc[:, sample_stats_columns]
    sample_df = df.loc[:, sample_columns]

    return sample_df, sample_stats, configuration_info, adaptation_info, timing_info

def _process_data_var(string):
    key, var = string.split("<-")
    if 'structure' in var:
        var, dim = var.replace("structure(", "").replace(",", "").split(".Dim")
        #dtype = int if '.' not in var and 'e' not in var.lower() else float
        dtype = float
        var = var.replace("c(", "").replace(")", "").strip().split()
        dim = dim.replace("=", "").replace("c(", "").replace(")", "").strip().split()
        dim = tuple(map(int, dim))
        var = np.fromiter(map(dtype, var), dtype).reshape(dim, order='F')
    elif 'c(' in var:
        #dtype = int if '.' not in var and 'e' not in var.lower() else float
        dtype = float
        var = var.replace("c(", "").replace(")", "").split(",")
        var = np.fromiter(map(dtype, var), dtype)
    else:
        #dtype = int if '.' not in var and 'e' not in var.lower() else float
        dtype = float
        var = dtype(var)
    return key.strip(), var

def read_data(path):
    data = {}
    with open(path, "r") as f_obj:
        var = ""
        for line in f_obj:
            if '<-' in line:
                if len(var):
                    key, var = _process_data_var(var)
                    data[key] = var
                var = ""
            var += " " + line.strip()
        if len(var):
            key, var = _process_data_var(var)
            data[key] = var
    return data

def unpack_dataframes(dfs):
    col_groups = defaultdict(list)
    columns = dfs[0].columns
    for col in columns:
        key, *loc = col.split('.')
        loc = list(map(int, loc))
        col_groups[key].append((col, loc))

    chains = len(dfs)
    draws = len(dfs[0])
    sample = {}
    for key, cols_locs in col_groups.items():
        ndim = np.array([loc for _, loc in cols_locs]).max(0)
        sample[key] = np.full((chains, draws, *ndim), np.nan)
        for col, loc in cols_locs:
            for chain_id, df in enumerate(dfs):
                draw = df[col].values
                if loc == []:
                    sample[key][chain_id, :] = draw
                else:
                    axis1_all = range(sample[key].shape[1])
                    loc = tuple([s-1 for s in loc])
                    slicer = (chain_id, axis1_all, *loc)
                    sample[key][slicer] = draw
    return sample

def from_cmdstan(*, output=None, prior=None, posterior_predictive=None,
                 observed_data=None, observed_data_var=None,
                 log_likelihood=None, coords=None, dims=None):
    """Convert CmdStan data into an InferenceData object.

    Parameters
    ----------
    output : List[Str]
        List of paths to output.csv files
    prior : List[Str]
        List of paths to output.csv files
    posterior_predictive : Str, List[Str]
        Posterior predictive samples for the fit. If endswith ".csv" assumes file.
    observed_data : Str
        Observed data used in the sampling. Path to data file in Rdump format.
    observed_data_var : Str, List[Str]
        Variable(s) used for slicing observed_data. If not defined, all
        data variables are imported.
    log_likelihood : Str, np.ndarray
        Pointwise log_likelihood for the data. If endswith ".csv" assumes file.
        A ndarray containing pointwise log_likelihood format:
            chain x draw x *data_shape
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable.

    Returns
    -------
    InferenceData object
    """
    return CmdStanConverter(
        output=output,
        prior=prior,
        posterior_predictive=posterior_predictive,
        observed_data=observed_data,
        observed_data_var=observed_data_var,
        log_likelihood=log_likelihood,
        coords=coords,
        dims=dims).to_inference_data()
