import numpy as np
import pandas as pd
import gzip
import lzma
import bz2
import os


__all__ = ['expand_variable_names', 'get_stats', 'get_varnames', 'log_post_trace',
           'trace_to_dataframe', 'save_trace', 'load_trace']


def expand_variable_names(trace, varnames):
    """
    Expand the name of variables to include multidimensional variables
    """
    tmp = []
    for vtrace in pd.unique(trace.columns):
        for v in varnames:
            if '{}__'.format(v) in vtrace or v in vtrace:
                tmp.append(vtrace)
    return np.unique(tmp)


def get_stats(trace, stat=None, combined=True):
    """
    get sampling statistics from trace

    Parameters
    ----------
    trace : Posterior sample
        Pandas DataFrame or PyMC3 trace
    stats : string
        Statistics
    combined : Bool
        If True multiple statistics from different chains will be combined together.
    Returns
    ----------
    stat: array with the choosen statistic
    """
    if type(trace).__name__ == 'MultiTrace':
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
    if varnames is None:
        return np.unique(trace.columns)
    else:
        return expand_variable_names(trace, varnames)


def log_post_trace(trace, model):
    """
    Calculate the elementwise log-posterior for the sampled trace.
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
    tr_t = type(trace).__name__
    mo_t = type(model).__name__

    if tr_t == 'MultiTrace' and mo_t == 'Model':
        cached = [(var, var.logp_elemwise) for var in model.observed_RVs]

        def logp_vals_point(pt):
            if len(model.observed_RVs) == 0:
                raise ValueError('The model does not contain observed values.')

            logp_vals = []
            for var, logp in cached:
                logp = logp(pt)
                if var.missing_values:
                    logp = logp[~var.observations.mask]
                logp_vals.append(logp.ravel())

            return np.concatenate(logp_vals)

        points = trace.points()
        logp = (logp_vals_point(pt) for pt in points)
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
    if type(trace).__name__ == 'MultiTrace':

        var_shapes = trace._straces[0].var_shapes
        varnames = [var for var in var_shapes.keys() if not _is_transformed_name(str(var))]

        flat_names = {v: _create_flat_names(v, var_shapes[v]) for v in varnames}

        var_dfs = []
        for v in varnames:
            vals = trace.get_values(v, combine=combined)
            if isinstance(vals, list):
                for va in vals:
                    flat_vals = va.reshape(va.shape[0], -1)
                    var_dfs.append(pd.DataFrame(flat_vals, columns=flat_names[v]))
            else:
                flat_vals = vals.reshape(vals.shape[0], -1)
                var_dfs.append(pd.DataFrame(flat_vals, columns=flat_names[v]))

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
    """
    Quickly check if a name was transformed with `get_transormed_name`

    Parameters
    ----------
    name : str
        Name to check

    Returns
    -------
    bool
        Boolean, whether the string could have been produced by `get_transormed_name`
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


def save_trace(trace, file_name='trace', compression='gzip', combined=False):
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
    trace = trace_to_dataframe(trace, combined=False)
    trace.to_csv('{}.{}'.format(file_name, compression), compression=compression)


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
    if ext == 'gzip':
        fd = gzip.open(filepath, 'rt')
    elif ext == 'bz2':
        fd = bz2.open(filepath, 'rt')
    elif ext == 'xz':
        fd = lzma.open(filepath, 'rt')
    else:
        fd = open(filepath)
    line = fd.readline().strip()
    fd.close()
    df.columns = [i for i in line.split(',') if i]
    if combined:
        df = trace_to_dataframe(df, combined)

    return df
