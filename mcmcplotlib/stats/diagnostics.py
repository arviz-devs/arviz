import numpy as np
from ..utils import trace_to_dataframe, get_varnames

__all__ = ['effective_n', 'gelman_rubin', 'geweke']


def effective_n(trace, varnames=None):
    R"""
    Returns estimate of the effective sample size of a set of traces.

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
      Posterior samples. at least 2 chains are needed to compute this diagnostic 
      of one or more stochastic parameters.
    varnames : list
      Names of variables to include in the effective_n report

    Returns
    -------
    n_eff : dictionary of floats (MultiTrace) or float (trace object)
        Return the effective sample size, :math:`\hat{n}_{eff}`

    Notes
    -----
    The diagnostic is computed by:

    .. math:: \hat{n}_{eff} = \frac{mn}{1 + 2 \sum_{t=1}^T \hat{\rho}_t}

    where :math:`\hat{\rho}_t` is the estimated _autocorrelation at lag t, and T
    is the first odd positive integer for which the sum
    :math:`\hat{\rho}_{T+1} + \hat{\rho}_{T+1}` is negative.

    The current implementation is similar to Stan, which uses Geyer's initial
    monotone sequence criterion (Geyer, 1992; Geyer, 2011).

    References
    ----------
    Gelman et al. BDA (2014)
    """

    trace = trace_to_dataframe(trace, combined=False)
    varnames = get_varnames(trace, varnames)

    if not np.all(trace.columns.duplicated(keep=False)):
        raise ValueError(
            'Calculation of effective sample size requires multiple chains of the same length.')
    else:
        n_eff = {}

        for var in varnames:
            n_eff[var] = _get_neff(trace[var].values.T)

        return n_eff


def _get_neff(trace_value):
    """Compute the effective sample size for a 2D array
    """
    nchain, n_samples = trace_value.shape

    acov = np.asarray([_autocov(trace_value[chain]) for chain in range(nchain)])

    chain_mean = trace_value.mean(axis=1)
    chain_var = acov[:, 0] * n_samples / (n_samples - 1.)
    acov_t = acov[:, 1] * n_samples / (n_samples - 1.)
    mean_var = np.mean(chain_var)
    var_plus = mean_var * (n_samples - 1.) / n_samples
    var_plus += np.var(chain_mean, ddof=1)

    rho_hat_t = np.zeros(n_samples)
    rho_hat_even = 1.
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1. - (mean_var - np.mean(acov_t)) / var_plus
    rho_hat_t[1] = rho_hat_odd
    # Geyer's initial positive sequence
    max_t = 1
    t = 1
    while t < (n_samples - 2) and (rho_hat_even + rho_hat_odd) >= 0.:
        rho_hat_even = 1. - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1. - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        max_t = t + 2
        t += 2

    # Geyer's initial monotone sequence
    t = 3
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2
    ess = (nchain * n_samples) / (-1. + 2. * np.sum(rho_hat_t))
    return ess


def _autocorr(x):
    """
    Compute autocorrelation using FFT for every lag for the input array
    https://en.wikipedia.org/wiki/autocorrelation#Efficient_computation

    Parameters
    ----------
    x : Numpy array
        An array containing MCMC samples

    Returns
    -------
    acorr: Numpy array same size as the input array
    """
    y = x - x.mean()
    n = len(y)
    result = fftconvolve(y, y[::-1])
    acorr = result[len(result) // 2:]
    acorr /= np.arange(n, 0, -1)
    acorr /= acorr[0]
    return acorr


def _autocov(x):
    """Compute autocovariance estimates for every lag for the input array

    Parameters
    ----------
    x : Numpy array
        An array containing MCMC samples

    Returns
    -------
    acov: Numpy array same size as the input array
    """
    acorr = _autocorr(x)
    varx = np.var(x, ddof=1) * (len(x) - 1) / len(x)
    acov = acorr * varx
    return acov


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
    varnames = get_varnames(trace, varnames)

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


def geweke(trace, varnames=None, first=.1, last=.5, intervals=20):
    R"""
    Return z-scores for convergence diagnostics.

    Compare the mean of the first % of series with the mean of the last % of
    series. x is divided into a number of segments for which this difference is
    computed. If the series is converged, this score should oscillate between
    -1 and 1.

    Parameters
    ----------
    x : array-like
      The trace of some stochastic parameter.
    first : float
      The fraction of series at the beginning of the trace.
    last : float
      The fraction of series at the end to be compared with the section
      at the beginning.
    intervals : int
      The number of segments.

    Returns
    -------
    scores : list [[]]
      Return a list of [i, score], where i is the starting index for each
      interval and score the Geweke score on the interval.

    Notes
    -----

    The Geweke score on some series x is computed by:

      .. math:: \frac{E[x_s] - E[x_e]}{\sqrt{V[x_s] + V[x_e]}}

    where :math:`E` stands for the mean, :math:`V` the variance,
    :math:`x_s` a section at the start of the series and
    :math:`x_e` a section at the end of the series.

    References
    ----------
    Geweke (1992)
    """

    trace = trace_to_dataframe(trace, combined=False)
    varnames = get_varnames(trace, varnames)

    gewekes = {}

    for var in varnames:
        gewekes[var] = _get_geweke(trace[var].values)

    return gewekes


def _get_geweke(x, first=.1, last=.5, intervals=20):
    # Filter out invalid intervals
    for interval in (first, last):
        if interval <= 0 or interval >= 1:
            raise ValueError(
                "Invalid intervals for Geweke convergence analysis",
                (first,
                 last))
    if first + last >= 1:
        raise ValueError(
            "Invalid intervals for Geweke convergence analysis",
            (first,
             last))

    # Initialize list of z-scores
    zscores = []

    # Last index value
    end = len(x) - 1

    # Start intervals going up to the <last>% of the chain
    last_start_idx = (1 - last) * end

    # Calculate starting indices
    start_indices = np.arange(0, int(last_start_idx), step=int(
        (last_start_idx) / (intervals - 1)))

    # Loop over start indices
    for start in start_indices:
        # Calculate slices
        first_slice = x[start: start + int(first * (end - start))]
        last_slice = x[int(end - last * (end - start)):]

        z = first_slice.mean() - last_slice.mean()
        z /= np.sqrt(first_slice.var() + last_slice.var())

        zscores.append([start, z])

    if intervals is None:
        return np.array(zscores[0])
    else:
        return np.array(zscores)
