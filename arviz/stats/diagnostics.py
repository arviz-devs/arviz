import warnings

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from ..utils import convert_to_xarray
from ..plots.plot_utils import xarray_var_iter, selection_to_string

__all__ = ['effective_n', 'gelman_rubin', 'geweke']


def effective_n(trace, var_names=None, round_to=2):
    R"""
    Returns estimate of the effective sample size of a set of traces.

    Parameters
    ----------
    posterior : xarray, or object that can be converted (pystan or pymc3 draws)
        Posterior samples. At least 2 chains are needed.
    var_names : list
      Names of variables to include in the effective_n report
    round_to : int
        Controls formatting for floating point numbers. Default 2.

    Returns
    -------
    n_eff : Pandas' DataFrame
        Return the effective sample size, :math:`\hat{n}_{eff}`

    Notes
    -----
    The diagnostic is computed by:

    .. math:: \hat{n}_{eff} = \frac{mn}{1 + 2 \sum_{t=1}^T \hat{\rho}_t}

    where :math:`\hat{\rho}_t` is the estimated _autocorrelation at lag t, and T
    is the first odd positive integer for which the sum
    :math:`\hat{\rho}_{T+1} + \hat{\rho}_{T+1}` is negative.

    The current implementation is similar to Stan, which uses Geyer's initial monotone sequence
    criterion (Geyer, 1992; Geyer, 2011).

    References
    ----------
    Gelman et al. BDA (2014)
    """
    data = convert_to_xarray(trace)

    if len(data.chain) < 2:
        raise ValueError('Calculation of effective sample size requires more than one chain')
    else:
        n_eff = []

        for var_name, selection, x in xarray_var_iter(data, var_names, True):
            n_eff.append((var_name, selection_to_string(selection), round(_get_neff(x), round_to)))

        return pd.DataFrame(n_eff,
                            columns=['var_name', 'dimensions', 'n_eff']).set_index('var_name')


def _get_neff(trace_value):
    """
    Compute the effective sample size for a 2D array
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


def autocorr(x):
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
    len_y = len(y)
    result = fftconvolve(y, y[::-1])
    acorr = result[len(result) // 2:]
    acorr /= np.arange(len_y, 0, -1)
    acorr /= acorr[0]
    return acorr


def _autocov(x):
    """
    Compute autocovariance estimates for every lag for the input array

    Parameters
    ----------
    x : Numpy array
        An array containing MCMC samples

    Returns
    -------
    acov: Numpy array same size as the input array
    """
    acorr = autocorr(x)
    varx = np.var(x, ddof=1) * (len(x) - 1) / len(x)
    acov = acorr * varx
    return acov


def gelman_rubin(trace, var_names=None, round_to=2):
    R"""
    Returns estimate of R for a set of traces.

    The Gelman-Rubin diagnostic tests for lack of convergence by comparing the variance between
    multiple chains to the variance within each chain. If convergence has been achieved, the
    between-chain and within-chain variances should be identical. To be most effective in
    detecting evidence for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    posterior : xarray, or object that can be converted (pystan or pymc3 draws)
        Posterior samples. At least 2 chains are needed.
    var_names : list
      Names of variables to include in the rhat report
    round_to : int
        Controls formatting for floating point numbers. Default 2.

    Returns
    -------
    r_hat : Pandas' DataFrame
      Returns dictionary of the potential scale reduction factors, :math:`\hat{R}`

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
    data = convert_to_xarray(trace)

    if len(data.chain) < 2:
        raise ValueError('Calculation of effective sample size requires more than one chain')
    else:
        r_hat = []

        for var_name, selection, x in xarray_var_iter(data, var_names, True):
            r_hat.append((var_name, selection_to_string(selection), round(_get_rhat(x), round_to)))

        return pd.DataFrame(r_hat,
                            columns=['var_name', 'dimensions', 'r_hat']).set_index('var_name')


def _get_rhat(values, round_to=2):
    """Compute the rhat for a 2d array
    """
    num_samples = values.shape[1]
    # Calculate between-chain variance
    between_chain_variance = num_samples * np.var(np.mean(values, axis=1), axis=0, ddof=1)
    # Calculate within-chain variance
    within_chain_variance = np.mean(np.var(values, axis=1, ddof=1), axis=0)
    # Estimate of marginal posterior variance
    v_hat = (within_chain_variance * (num_samples - 1) / num_samples +
             between_chain_variance / num_samples)

    return round((v_hat / within_chain_variance)**0.5, round_to)


def geweke(trace, var_names=None, first=.1, last=.5, intervals=20):
    R"""
    Return z-scores for convergence diagnostics.

    Compare the mean of the first % of series with the mean of the last % of series. x is divided
    into a number of segments for which this difference is computed. If the series is converged,
    this score should oscillate between -1 and 1.

    Parameters
    ----------
    posterior : xarray, or object that can be converted (pystan or pymc3 draws)
    var_names : list
      Names of variables to include in the rhat report
    first : float
      The fraction of series at the beginning of the trace.
    last : float
      The fraction of series at the end to be compared with the section
      at the beginning.
    intervals : int
      The number of segments.

    Returns
    -------
    scores : Pandas's DataFrame
      The r_hat column contains lists of [i, score], where i is the starting index for each
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
    data = convert_to_xarray(trace)

    geweke = []
    for var_name, selection, x in xarray_var_iter(data, var_names, False):
        geweke.append((var_name, selection_to_string(selection), 
                     _get_geweke(x, first, last, intervals)))

    return pd.DataFrame(geweke, columns=['var_name', 'dimensions', 'r_hat']).set_index('var_name')


def _get_geweke(x, first=.1, last=.5, intervals=20):
    # Filter out invalid intervals
    for interval in (first, last):
        if interval <= 0 or interval >= 1:
            raise ValueError("Invalid intervals for Geweke convergence analysis", (first, last))
    if first + last >= 1:
        raise ValueError("Invalid intervals for Geweke convergence analysis", (first, last))

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

        z_score = first_slice.mean() - last_slice.mean()
        z_score /= np.sqrt(first_slice.var() + last_slice.var())

        zscores.append([start, z_score])

    if intervals is None:
        return np.array(zscores[0])
    else:
        return np.array(zscores)


def ks_summary(pareto_tail_indices):
    """
    Display a summary of Pareto tail indices.

    Parameters
    ----------
    pareto_tail_indices : array
      Pareto tail indices.

    Returns
    -------
    df_k : dataframe
      Dataframe containing k diagnostic values.
    """
    kcounts, _ = np.histogram(pareto_tail_indices, bins=[-np.Inf, .5, .7, 1, np.Inf])
    kprop = kcounts/len(pareto_tail_indices)*100
    df_k = (pd.DataFrame(dict(_=['(good)', '(ok)', '(bad)', '(very bad)'],
                              Count=kcounts,
                              Pct=kprop))
            .rename(index={0: '(-Inf, 0.5]',
                           1: ' (0.5, 0.7]',
                           2: '   (0.7, 1]',
                           3: '   (1, Inf)'}))

    if np.sum(kcounts[1:]) == 0:
        warnings.warn("All Pareto k estimates are good (k < 0.5)")
    elif np.sum(kcounts[2:]) == 0:
        warnings.warn("All Pareto k estimates are ok (k < 0.7)")

    return df_k
