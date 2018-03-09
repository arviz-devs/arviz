import numpy as np
import pandas as pd
from collections import namedtuple
from ..utils import trace_to_dataframe


def hpd(x, alpha=0.05, transform=lambda x: x):
    """
    Calculate highest posterior density (HPD) of array for given alpha. The HPD is the minimum
    width Bayesian credible interval (BCI).

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

        return np.array(_calc_min_interval(sx, alpha))


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
    Internal method to determine the minimum interval of a given width. Assumes that x is a
    sorted numpy array.
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


def gelman_rubin(trace, varnames=None, round_to=2):
    R"""Returns estimate of R for a set of traces.

    The Gelman-Rubin diagnostic tests for lack of convergence by comparing the variance between
    multiple chains to the variance within each chain. If convergence has been achieved, the
    between-chain and within-chain variances should be identical. To be most effective in
    detecting evidence for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples. at least 2 chains are needed to compute this diagnostic 
    varnames : list
      Names of variables to include in the rhat report
    round_to : int
        Controls formatting for floating point numbers. Default 2.

    Returns
    -------
    Rhat : dict of floats (MultiTrace) or float (trace object)
      Returns dictionary of the potential scale reduction
      factors, :math:`\hat{R}`

    Notes
    -----

    The diagnostic is computed by:

      .. math:: \hat{R} = \frac{\hat{V}}{W}

    where :math:`W` is the within-chain variance and :math:`\hat{V}` is the posterior variance
    estimate for the pooled traces. This is the potential scale reduction factor, which converges
    to unity when each of the traces is a sample from the target posterior. Values greater than one
    indicate that one or more chains have not yet converged.

    References
    ----------
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)
    """

    trace = trace_to_dataframe(trace, combined=False)

    if varnames is None:
        varnames = pd.unique(trace.columns)
    else:
        varnames = expand_variable_names(trace, varnames)

    if not np.all(trace.columns.duplicated(keep=False)):
        raise ValueError('Gelman-Rubin diagnostic requires multiple chains of the same length.')
    else:
        Rhat = {}

        for var in varnames:
            x = trace[var].values.T
            num_samples = x.shape[1]
            # Calculate between-chain variance
            B = num_samples * np.var(np.mean(x, axis=1), axis=0, ddof=1)
            # Calculate within-chain variance
            W = np.mean(np.var(x, axis=1, ddof=1), axis=0)
            # Estimate of marginal posterior variance
            Vhat = W * (num_samples - 1) / num_samples + B / num_samples

            Rhat[var] = round((Vhat / W)**0.5, round_to)

        return Rhat


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
