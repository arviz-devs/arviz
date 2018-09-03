"""Miscellaneous utilities for supporting ArviZ."""
import bz2
import gzip
import importlib
import lzma
import os
import re

import numpy as np
import pandas as pd

from ..inference_data import InferenceData


def _has_type(object_, typename, module_path):
    """Check if an object is an instance of a type under a given module.

    Avoids explicit dependencies on a module found under `module_path`
    by using `importlib`.

    Parameters
    ----------
    object_ : object
        Arbitrary python object.
    typename : str
        Name of a type in the module under `module_path`.
    module_path : str
        Import path to a module that contains `typename`.
        Is fed to `importlib.import_module`.

    Returns
    -------
    has_type : bool
        `True` if `isinstance(object, getattr(importlib.import_module(module_path), typename))`.
        `False` if this condition does not hold or the module under
        `module_path` is not installed.

    Examples
    --------
    A string is not a real number:

    >>> _has_type("aha", typename="Real", module_path="numbers")
    False

    A timedelta object has type timedelta:

    >>> from datetime import timedelta
    >>> object_ = timedelta(10)
    >>> _has_type(object_, typename="timedelta", module_path="datetime")
    True
    """
    try:
        type_cls = getattr(importlib.import_module(module_path), typename)
    except ImportError:
        return False
    else:
        return isinstance(object_, type_cls)


def untransform_varnames(varnames):
    """Map transformed variable names back to their originals.

    Mainly useful for dealing with PyMC3 traces.

    Example
    -------
    untransform_varnames(['eta__0', 'eta__1', 'theta', 'theta_log__'])

    {'eta': {'eta__0', 'eta_1'}, 'theta': {'theta'}}, {'theta': {'theta_log__'}}

    Parameters
    ----------
    varnames : iterable of strings
        All the varnames from a trace

    Returns
    -------
    (dict, dict)
        A dictionary of names to vector names, and names to transformed names
    """
    # Captures tau_log____0 or tau_log__, but not tau__0
    transformed_vec_ptrn = re.compile(r'^(.*)__(?:__\d+)$')
    # Captures tau__0 and tau_log____0, so use after the above
    vec_ptrn = re.compile(r'^(.*)__\d+$')

    varname_map = {}
    transformed = {}
    for varname in varnames:
        has_match = False
        for ptrn, mapper in ((transformed_vec_ptrn, transformed), (vec_ptrn, varname_map)):
            match = ptrn.match(varname)
            if match:
                base_name = match.group(1)
                if base_name not in mapper:
                    mapper[base_name] = set()
                mapper[base_name].add(varname)
                has_match = True
        if not has_match:
            if varname not in varname_map:
                varname_map[varname] = set()
            varname_map[varname].add(varname)
    return varname_map, transformed


def expand_variable_names(trace, varnames):
    """Expand the name of variables to include multidimensional variables."""
    tmp = []
    for vtrace in pd.unique(trace.columns):
        for varname in varnames:
            if vtrace == varname or vtrace.startswith('{}__'.format(varname)):
                tmp.append(vtrace)
    return np.unique(tmp)


def get_stats(trace, stat=None, combined=True):
    """Get sampling statistics from trace.

    Parameters
    ----------
    trace : Posterior sample
        Pandas DataFrame or PyMC3 trace
    stat : string
        Statistics
    combined : Bool
        If True multiple statistics from different chains will be combined together.

    Returns
    -------
    stat: array with the chosen statistic
    """
    if _has_type(trace, typename="MultiTrace", module_path="pymc3.backends.base"):
        try:
            return trace.get_sampler_stats(stat, combine=combined)
        except KeyError:
            print('There is no {} information in the passed trace.'.format(stat))

    elif isinstance(trace, pd.DataFrame):
        try:
            return trace[stat].values
        except KeyError:
            print('There is no {} information in the passed trace.'.format(stat))

    else:
        raise ValueError('The trace should be a DataFrame or a trace from PyMC3')


def get_varnames(trace, varnames):
    """Extract variable names from a trace."""
    if varnames is None:
        return np.unique(trace.columns)
    else:
        return expand_variable_names(trace, varnames)


def log_post_trace(trace, model):
    """Calculate the elementwise log-posterior for the sampled trace.

    Currently only supports trace and models from PyMC3.

    Parameters
    ----------
    trace : trace object
        Posterior samples
    model : PyMC Model

    Returns
    -------
    logp : array of shape (n_samples, n_observations)
        The contribution of the observations to the logp of the whole model.
    """
    is_pymc3_multitrace = _has_type(
        object_=trace, typename="MultiTrace", module_path="pymc3.backends.base"
    )

    is_pymc3_model = _has_type(object_=model, typename="Model", module_path="pymc3")

    if is_pymc3_multitrace and is_pymc3_model:
        cached = [(var, var.logp_elemwise) for var in model.observed_RVs]

        def logp_vals_point(point):
            if len(model.observed_RVs) == 0:
                raise ValueError('The model does not contain observed values.')

            logp_vals = []
            for var, logp in cached:
                logp = logp(point)
                if var.missing_values:
                    logp = logp[~var.observations.mask]
                logp_vals.append(logp.ravel())

            return np.concatenate(logp_vals)

        points = trace.points()
        logp = (logp_vals_point(point) for point in points)
        return np.stack(logp)
    else:
        raise ValueError('Currently only supports trace and models from PyMC3.')


def trace_to_dataframe(trace, combined=True):
    """Convert trace to Pandas DataFrame.

    Parameters
    ----------
    trace : trace
        PyMC3's trace or Pandas DataFrame
    combined : Bool
        If True multiple chains will be combined together in the same columns. Otherwise they will
        be assigned to separate columns.
    """
    if _has_type(object_=trace,
                 typename="MultiTrace",
                 module_path="pymc3.backends.base"):
        var_shapes = trace._straces[0].var_shapes  # pylint: disable=protected-access
        varnames = [var for var in var_shapes.keys() if not _is_transformed_name(str(var))]

        flat_names = {v: _create_flat_names(v, var_shapes[v]) for v in varnames}

        var_dfs = []
        for varname in varnames:
            vals = trace.get_values(varname, combine=combined)
            if isinstance(vals, list):
                for val in vals:
                    flat_vals = val.reshape(val.shape[0], -1)
                    var_dfs.append(pd.DataFrame(flat_vals, columns=flat_names[varname]))
            else:
                flat_vals = vals.reshape(vals.shape[0], -1)
                var_dfs.append(pd.DataFrame(flat_vals, columns=flat_names[varname]))

    elif isinstance(trace, pd.DataFrame):
        if combined:
            varnames = get_varnames(trace, trace.columns)
            trace = pd.DataFrame({v: trace[v].values.ravel() for v in varnames})
        return trace

    else:
        raise ValueError('The trace should be a DataFrame or a trace from PyMC3')

    return pd.concat(var_dfs, axis=1)


def _create_flat_names(varname, shape):
    """Return flat variable names for `varname` of `shape`.

    Examples
    --------
    >>> create_flat_names('x', (5,))
    ['x__0', 'x__1', 'x__2', 'x__3', 'x__4']
    >>> create_flat_names('x', (2, 2))
    ['x__0_0', 'x__0_1', 'x__1_0', 'x__1_1']
    """
    if not shape:
        return [varname]
    labels = (np.ravel(xs).tolist() for xs in np.indices(shape))
    labels = (map(str, xs) for xs in labels)
    return ['{}__{}'.format(varname, '_'.join(idxs)) for idxs in zip(*labels)]


def _is_transformed_name(name):
    """Quickly check if a name was transformed with `get_transformed_name`.

    Parameters
    ----------
    name : str
        Name to check

    Returns
    -------
    bool
        Boolean, whether the string could have been produced by `get_transformed_name`
    """
    return name.endswith('__') and name.count('_') >= 3


def _create_flat_names(varname, shape):
    """
    Return flat variable names for `varname` of `shape`.

    Examples
    --------
    >>> create_flat_names('x', (5,))
    ['x__0', 'x__1', 'x__2', 'x__3', 'x__4']
    >>> create_flat_names('x', (2, 2))
    ['x__0_0', 'x__0_1', 'x__1_0', 'x__1_1']
    """
    if not shape:
        return [varname]
    labels = (np.ravel(xs).tolist() for xs in np.indices(shape))
    labels = (map(str, xs) for xs in labels)
    return ['{}__{}'.format(varname, '_'.join(idxs)) for idxs in zip(*labels)]


def save_trace(trace, filename='trace.gzip', compression='gzip', combined=False):
    """
    Save trace to a csv file. Duplicated columns names will be preserved, if any.

    Parameters
    ----------
    trace : trace
        PyMC3's trace or Pandas DataFrame
    filepath : str
        name or path of the file to save trace
    compression : str, optional
        String representing the compression to use in the output file, allowed values are
        'gzip' (default), 'bz2' and 'xz'.
    combined : Bool
        If True multiple chains will be combined together in the same columns. Otherwise they will
        be assigned to separate columns. Defaults to False
    """
    trace = trace_to_dataframe(trace, combined=combined)
    trace.to_csv(filename, compression=compression)


def load_trace(filepath, combined=False):
    """
    Load csv file into a DataFrame. Duplicated columns names will be preserved, if any.

    Parameters
    ----------
    filepath : str
        name or path of the file to save trace
    combined : Bool
        If True multiple chains will be combined together in the same columns. Otherwise they will
        be assigned to separate columns. Defaults to False
    """
    ext = os.path.splitext(filepath)[1][1:]
    df = pd.read_csv(filepath, index_col=0, compression=ext)
    try:
        if ext == 'gzip':
            file_descriptor = gzip.open(filepath, 'rt')
        elif ext == 'bz2':
            file_descriptor = bz2.open(filepath, 'rt')
        elif ext == 'xz':
            file_descriptor = lzma.open(filepath, 'rt')
        else:
            file_descriptor = open(filepath)
        line = file_descriptor.readline().strip()
    finally:
        file_descriptor.close()
    df.columns = [i for i in line.split(',') if i]
    if combined:
        df = trace_to_dataframe(df, combined)

    return df


def load_data(filename):
    """Load netcdf file back into an arviz.InferenceData.

    Parameters
    ----------
    filename : str
        name or path of the file to load trace
    """
    return InferenceData(filename)


def load_arviz_data(dataset):
    """Load built-in arviz dataset into memory.

    Will print out available datasets in case of error.

    Parameters
    ----------
    dataset : str
        Name of dataset to load

    Returns
    -------
    InferenceData
    """
    top = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(top, 'doc', 'data')
    datasets_available = {
        'centered_eight': {
            'description': '''
                Centered eight schools model.  Four chains, 500 draws each, fit with
                NUTS in PyMC3.  Features named coordinates for each of the eight schools.
            ''',
            'path': os.path.join(data_path, 'centered_eight.nc')
        },
        'non_centered_eight': {
            'description': '''
                Non-centered eight schools model.  Four chains, 500 draws each, fit with
                NUTS in PyMC3.  Features named coordinates for each of the eight schools.
            ''',
            'path': os.path.join(data_path, 'non_centered_eight.nc')
        }
    }
    if dataset in datasets_available:
        return InferenceData.from_netcdf(datasets_available[dataset]['path'])
    else:
        msg = ['\'dataset\' must be one of the following options:']
        for key, value in sorted(datasets_available.items()):
            msg.append('{key}: {description}'.format(key=key, description=value['description']))

        raise ValueError('\n'.join(msg))
