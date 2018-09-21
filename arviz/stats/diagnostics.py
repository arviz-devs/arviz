"""Diagnostic functions for ArviZ."""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
import xarray as xr

from arviz import convert_to_dataset


__all__ = ['effective_n', 'gelman_rubin', 'geweke', 'autocorr']


def effective_n(data, *, var_names=None):
    r"""Calculate estimate of the effective sample size.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
    var_names : list
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

    The current implementation is similar to Stan, which uses Geyer's initial monotone sequence
    criterion (Geyer, 1992; Geyer, 2011).

    References
    ----------
    Gelman et al. BDA (2014)
    """
    if isinstance(data, np.ndarray):
        return _get_neff(data)

    dataset = convert_to_dataset(data, group='posterior')
    if var_names is None:
        var_names = list(dataset.data_vars)
    dataset = dataset[var_names]
    return xr.apply_ufunc(_neff_ufunc, dataset, input_core_dims=(('chain', 'draw',),))


def _get_neff(sample_array):
    """Compute the effective sample size for a 2D array."""
    shape = sample_array.shape
    if len(shape) != 2:
        raise TypeError('Effective sample size calculation requires 2 dimensional arrays.')
    n_chain, n_draws = shape
    if n_chain <= 1:
        raise TypeError('Effective sample size calculation requires multiple chains.')

    acov = np.asarray([_autocov(sample_array[chain]) for chain in range(n_chain)])

    chain_mean = sample_array.mean(axis=1)
    chain_var = acov[:, 0] * n_draws / (n_draws - 1.)
    acov_t = acov[:, 1] * n_draws / (n_draws - 1.)
    mean_var = np.mean(chain_var)
    var_plus = mean_var * (n_draws - 1.) / n_draws
    var_plus += np.var(chain_mean, ddof=1)

    rho_hat_t = np.zeros(n_draws)
    rho_hat_even = 1.
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1. - (mean_var - np.mean(acov_t)) / var_plus
    rho_hat_t[1] = rho_hat_odd
    # Geyer's initial positive sequence
    max_t = 1
    t = 1
    while t < (n_draws - 2) and (rho_hat_even + rho_hat_odd) >= 0.:
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
    ess = (n_chain * n_draws) / (-1. + 2. * np.sum(rho_hat_t))
    return ess


def _neff_ufunc(ary):
    """Ufunc for computing effective sample size.

    This can be used on an xarray Dataset, using
    `xr.apply_ufunc(_neff_ufunc, ..., input_core_dims=(('chain', 'draw'),))
    """
    target = np.empty(ary.shape[:-2])
    for idxs in itertools.product(*[np.arange(d) for d in target.shape]):
        idxs = list(idxs)
        idxs.append(Ellipsis)
        target[idxs] = _get_neff(ary[idxs])
    return target


def autocorr(x):
    """Compute autocorrelation using FFT for every lag for the input array.

    See https://en.wikipedia.org/wiki/autocorrelation#Efficient_computation

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
    """Compute autocovariance estimates for every lag for the input array.

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


def gelman_rubin(data, var_names=None):
    r"""Compute estimate of R-hat for a set of traces.

    The Gelman-Rubin diagnostic tests for lack of convergence by comparing the variance between
    multiple chains to the variance within each chain. If convergence has been achieved, the
    between-chain and within-chain variances should be identical. To be most effective in
    detecting evidence for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
    var_names : list
      Names of variables to include in the rhat report

    Returns
    -------
    r_hat : dict of floats (MultiTrace) or float (trace object)
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
    if isinstance(data, np.ndarray):
        return _get_rhat(data)

    dataset = convert_to_dataset(data, group='posterior')
    if var_names is None:
        var_names = list(dataset.data_vars)
    dataset = dataset[var_names]
    return xr.apply_ufunc(_rhat_ufunc, dataset, input_core_dims=(('chain', 'draw',),))


def _get_rhat(values, round_to=2):
    """Compute the rhat for a 2d array."""
    shape = values.shape
    if len(shape) != 2:
        raise TypeError('Effective sample size calculation requires 2 dimensional arrays.')
    _, num_samples = shape

    # Calculate between-chain variance
    between_chain_variance = num_samples * np.var(np.mean(values, axis=1), axis=0, ddof=1)
    # Calculate within-chain variance
    within_chain_variance = np.mean(np.var(values, axis=1, ddof=1), axis=0)
    # Estimate of marginal posterior variance
    v_hat = (within_chain_variance * (num_samples - 1) / num_samples +
             between_chain_variance / num_samples)

    return round((v_hat / within_chain_variance)**0.5, round_to)

def _rhat_ufunc(ary):
    """Ufunc for computing effective sample size.

    This can be used on an xarray Dataset, using
    `xr.apply_ufunc(_neff_ufunc, ..., input_core_dims=(('chain', 'draw'),))
    """
    target = np.empty(ary.shape[:-2])
    for idxs in itertools.product(*[np.arange(d) for d in target.shape]):
        idxs = list(idxs)
        idxs.append(Ellipsis)
        target[idxs] = _get_rhat(ary[idxs])
    return target


def geweke(values, first=.1, last=.5, intervals=20):
    r"""Compute z-scores for convergence diagnostics.

    Compare the mean of the first % of series with the mean of the last % of series. x is divided
    into a number of segments for which this difference is computed. If the series is converged,
    this score should oscillate between -1 and 1.

    Parameters
    ----------
    values : 1D array-like
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
      Return a list of [i, score], where i is the starting index for each interval and score the
      Geweke score on the interval.

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
    # Filter out invalid intervals
    for interval in (first, last):
        if interval <= 0 or interval >= 1:
            raise ValueError("Invalid intervals for Geweke convergence analysis", (first, last))
    if first + last >= 1:
        raise ValueError("Invalid intervals for Geweke convergence analysis", (first, last))

    # Initialize list of z-scores
    zscores = []

    # Last index value
    end = len(values) - 1

    # Start intervals going up to the <last>% of the chain
    last_start_idx = (1 - last) * end

    # Calculate starting indices
    start_indices = np.linspace(0, last_start_idx, num=intervals, endpoint=True, dtype=int)

    # Loop over start indices
    for start in start_indices:
        # Calculate slices
        first_slice = values[start: start + int(first * (end - start))]
        last_slice = values[int(end - last * (end - start)):]

        z_score = first_slice.mean() - last_slice.mean()
        z_score /= np.sqrt(first_slice.var() + last_slice.var())

        zscores.append([start, z_score])

    if intervals is None:
        return np.array(zscores[0])
    else:
        return np.array(zscores)


def ks_summary(pareto_tail_indices):
    """Display a summary of Pareto tail indices.

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
