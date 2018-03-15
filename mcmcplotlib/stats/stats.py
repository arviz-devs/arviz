import numpy as np
import pandas as pd
from collections import namedtuple
from ..utils import trace_to_dataframe, get_varnames, _create_flat_names
from .diagnostics import effective_n, gelman_rubin


__all__ = ['hpd', 'r2_score', 'summary']


def hpd(x, alpha=0.05, transform=lambda x: x):
    """
    Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI).

    Parameters
    ----------
    x : Numpy array
        An array containing posterior samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    transform : callable
        Function to transform data (defaults to identity)
    """
    # Make a copy of trace
    x = transform(x.copy())

    # For multivariate node
    if x.ndim > 1:

        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
        dims = np.shape(tx)

        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1] + (2,))

        for index in _make_indices(dims[:-1]):

            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])

            # Append to list
            intervals[index] = _calc_min_interval(sx, alpha)

        # Transpose back before returning
        return np.array(intervals)

    else:
        # Sort univariate node
        sx = np.sort(x)

        return _calc_min_interval(sx, alpha)


def _make_indices(dimensions):
    """ 
    Generates complete set of indices for given dimensions
    """
    level = len(dimensions)
    if level == 1:
        return list(range(dimensions[0]))
    indices = [[]]
    while level:
        _indices = []
        for j in range(dimensions[level - 1]):
            _indices += [[j] + i for i in indices]
        indices = _indices
        level -= 1
    try:
        return [tuple(i) for i in indices]
    except TypeError:
        return indices


def _calc_min_interval(x, alpha):
    """
    Internal method to determine the minimum interval of a given width. 
    Assumes that x is a sorted numpy array.
    """
    n = len(x)
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return hdi_min, hdi_max


def _hpd_df(x, alpha):
    cnames = ['hpd_{0:g}'.format(100 * alpha / 2),
              'hpd_{0:g}'.format(100 * (1 - alpha / 2))]
    return pd.DataFrame(hpd(x, alpha), columns=cnames)


def r2_score(y_true, y_pred, round_to=2):
    """
    R-squared for Bayesian regression models. Only valid for linear models.
    http://www.stat.columbia.edu/%7Egelman/research/unpublished/bayes_R2.pdf

    Parameters
    ----------
    y_true: : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    round_to : int
        Number of decimals used to round results (default 2).

    Returns
    -------
    `namedtuple` with the following elements:
    R2_median: median of the Bayesian R2
    R2_mean: mean of the Bayesian R2
    R2_std: standard deviation of the Bayesian R2
    """
    dimension = None
    if y_true.ndim > 1:
        dimension = 1

    var_y_est = np.var(y_pred, axis=dimension)
    var_e = np.var(y_true - y_pred, axis=dimension)

    r2 = var_y_est / (var_y_est + var_e)
    r2_median = np.around(np.median(r2), round_to)
    r2_mean = np.around(np.mean(r2), round_to)
    r2_std = np.around(np.std(r2), round_to)
    r2_r = namedtuple('r2_r', 'r2_median, r2_mean, r2_std')
    return r2_r(r2_median, r2_mean, r2_std)


def summary(trace, varnames=None, round_to=2, transform=lambda x: x,
            stat_funcs=None, extend=False, alpha=0.05, skip_first=0, batches=None):
    R"""
    Create a data frame with summary statistics.

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list
        Names of variables to include in summary
    round_to : int
        Controls formatting for floating point numbers. Default 2.
    transform : callable
        Function to transform data (defaults to identity)
    stat_funcs : None or list
        A list of functions used to calculate statistics. By default,
        the mean, standard deviation, simulation standard error, and
        highest posterior density intervals are included.

        The functions will be given one argument, the samples for a
        variable as a 2 dimensional array, where the first axis
        corresponds to sampling iterations and the second axis
        represents the flattened variable (e.g., x__0, x__1,...). Each
        function should return either

        1) A `pandas.Series` instance containing the result of
           calculating the statistic along the first axis. The name
           attribute will be taken as the name of the statistic.
        2) A `pandas.DataFrame` where each column contains the
           result of calculating the statistic along the first axis.
           The column names will be taken as the names of the
           statistics.
    extend : boolean
        If True, use the statistics returned by `stat_funcs` in
        addition to, rather than in place of, the default statistics.
        This is only meaningful when `stat_funcs` is not None.
    include_transformed : bool
        Flag for reporting automatically transformed variables in addition
        to original variables (defaults to False).
    alpha : float
        The alpha level for generating posterior intervals. Defaults
        to 0.05. This is only meaningful when `stat_funcs` is None.
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    batches : None or int
        Batch size for calculating standard deviation for non-independent
        samples. Defaults to the smaller of 100 or the number of samples.
        This is only meaningful when `stat_funcs` is None.

    Returns
    -------
    `pandas.DataFrame` with summary statistics for each variable Defaults one
    are: `mean`, `sd`, `mc_error`, `hpd_2.5`, `hpd_97.5`, `n_eff` and `Rhat`.
    Last two are only computed for traces with 2 or more chains.

    Examples
    --------
    .. code:: ipython
        >>> pm.summary(trace, ['mu'])
                   mean        sd  mc_error     hpd_5    hpd_95
        mu__0  0.106897  0.066473  0.001818 -0.020612  0.231626
        mu__1 -0.046597  0.067513  0.002048 -0.174753  0.081924

                  n_eff      Rhat
        mu__0     487.0   1.00001
        mu__1     379.0   1.00203

    Other statistics can be calculated by passing a list of functions.

    .. code:: ipython

        >>> import pandas as pd
        >>> def trace_sd(x):
        ...     return pd.Series(np.std(x, 0), name='sd')
        ...
        >>> def trace_quantiles(x):
        ...     return pd.DataFrame(pm.quantiles(x, [5, 50, 95]))
        ...
        >>> pm.summary(trace, ['mu'], stat_funcs=[trace_sd, trace_quantiles])
                     sd         5        50        95
        mu__0  0.066473  0.000312  0.105039  0.214242
        mu__1  0.067513 -0.159097 -0.045637  0.062912
    """
    trace = trace_to_dataframe(trace, combined=False)[skip_first:]
    varnames = get_varnames(trace, varnames)

    if batches is None:
        batches = min([100, len(trace)])

    cnames = ['hpd_{0:g}'.format(100 * alpha / 2),
              'hpd_{0:g}'.format(100 * (1 - alpha / 2))]

    funcs = [lambda x: pd.Series(np.mean(x, 0), name='mean').round(round_to),
             lambda x: pd.Series(np.std(x, 0), name='sd').round(round_to),
             lambda x: pd.Series(_mc_error(x, batches).round(round_to), name='mc_error'),
             lambda x: pd.DataFrame([hpd(x, alpha)], columns=cnames).round(round_to)]

    if stat_funcs is not None:
        if extend:
            funcs = funcs + stat_funcs
        else:
            funcs = stat_funcs

    var_dfs = []
    for var in varnames:
        vals = np.ravel(trace[var].values)
        var_df = pd.concat([f(vals) for f in funcs], axis=1)
        var_df.index = [var]
        var_dfs.append(var_df)
    dforg = pd.concat(var_dfs, axis=0)

    if (stat_funcs is not None) and (not extend):
        return dforg
    elif trace.columns.value_counts()[0] < 2:
        return dforg
    else:
        n_eff = effective_n(trace, varnames=varnames, round_to=round_to)
        rhat = gelman_rubin(trace, varnames=varnames, round_to=round_to)
        return pd.concat([dforg, n_eff, rhat], axis=1, join_axes=[dforg.index])


def _mc_error(x, batches=5):
    R"""Calculates the simulation standard error, accounting for non-independent
        samples. The trace is divided into batches, and the standard deviation of
        the batch means is calculated.

    Parameters
    ----------
    x : Numpy array
        An array containing MCMC samples
    batches : integer
        Number of batches

    Returns
    -------
    `float` representing the error
    """
    if x.ndim > 1:

        dims = np.shape(x)
        #ttrace = np.transpose(np.reshape(trace, (dims[0], sum(dims[1:]))))
        trace = np.transpose([t.ravel() for t in x])

        return np.reshape([_mc_error(t, batches) for t in trace], dims[1:])

    else:
        if batches == 1:
            return np.std(x) / np.sqrt(len(x))

        try:
            batched_traces = np.resize(x, (batches, int(len(x) / batches)))
        except ValueError:
            # If batches do not divide evenly, trim excess samples
            resid = len(x) % batches
            new_shape = (batches, (len(x) - resid) / batches)
            batched_traces = np.resize(x[:-resid], new_shape)

        means = np.mean(batched_traces, 1)

        return np.std(means) / np.sqrt(batches)
