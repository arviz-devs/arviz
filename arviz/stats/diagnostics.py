"""Diagnostic functions for ArviZ."""
import warnings

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
import xarray as xr

from ..data import convert_to_dataset
from ..utils import _var_names


__all__ = ["effective_sample_size", "rhat", "geweke", "autocorr"]


def effective_sample_size(data, *, var_names=None):
    r"""Calculate estimate of the effective sample size.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list
      Names of variables to include in the effective_sample_size report

    Returns
    -------
    ess : xarray.Dataset
        Return the effective sample size, :math:`\hat{N}_{eff}`

    Notes
    -----
    The diagnostic is computed by:

    .. math:: \hat{N}_{eff} = \frac{MN}{\hat{\tau}}
    .. math:: \hat{\tau} = -1 + 2 \sum_{t'=0}^K \hat{P}_t'

    where :math:`\hat{\rho}_t` is the estimated _autocorrelation at lag t, and T
    is the first odd positive integer for which the sum
    :math:`\hat{\rho}_{T+1} + \hat{\rho}_{T+1}` is negative.

    The current implementation is similar to Stan, which uses Geyer's initial monotone sequence
    criterion (Geyer, 1992; Geyer, 2011).

    References
    ----------
    https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html Section 15.4.2

    Gelman et al. BDA (2014) Formula 11.8
    """
    if isinstance(data, np.ndarray):
        return _get_ess(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    return xr.apply_ufunc(_ess_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def _ess_ufunc(ary):
    """Ufunc for computing effective sample size.

    This can be used on an xarray Dataset, using
    `xr.apply_ufunc(_ess_ufunc, ..., input_core_dims=(('chain', 'draw'),))
    """
    return _get_ess(ary)


def _get_ess(sample_array):
    """Compute the effective sample size for a ND array.

    It is required that the last two dimensions are chain dimension and draw dimension.
    """
    # pylint: disable=no-member
    if sample_array.ndim == 1:
        sample_array = sample_array[None, ...]
    n_chain, n_draws = sample_array.shape[-2], sample_array.shape[-1]

    acov = _autocov(sample_array)

    chain_mean = sample_array.mean(axis=-1, keepdims=True)
    chain_var = acov[..., :1] * n_draws / (n_draws - 1.0)
    mean_var = np.mean(chain_var, axis=-2)
    var_plus = mean_var * (n_draws - 1.0) / n_draws
    if n_chain > 1:
        var_plus += np.var(chain_mean, axis=-2, ddof=1)
    else:
        # to make rho_hat_t = autocorr(sample_array)
        mean_var = var_plus

    # Geyer's initial positive sequence
    rho_hat_t = 1.0 - (mean_var - acov.mean(axis=-2)) / var_plus
    rho_hat_t[..., 0] = 1.0  # correlation at lag 0 is 1
    # take sum of even index and odd index from the sequence
    p_t = rho_hat_t[..., :-1:2] + rho_hat_t[..., 1::2]

    # Geyer's initial monotone sequence
    # here we split out the initial value and take the accumulated min of the remaining sequence
    p_t = np.concatenate(
        [p_t[..., :1], np.minimum.accumulate(p_t[..., 1:].clip(min=0), axis=-1)],
        axis=-1
    )

    ess = np.floor((n_chain * n_draws) / (-1.0 + 2.0 * np.sum(p_t, axis=-1)))
    return ess


def autocorr(x):
    """Compute autocorrelation (over the last axis) using FFT for every lag for the input array.

    See https://en.wikipedia.org/wiki/autocorrelation#Efficient_computation

    Parameters
    ----------
    x : Numpy array
        An array containing MCMC samples

    Returns
    -------
    acorr: Numpy array same size as the input array
    """
    y = x - x.mean(axis=-1, keepdims=True)
    len_y = y.shape[-1]
    result = fftconvolve(y, y[..., ::-1], axes=-1)
    acorr = result[..., result.shape[-1] // 2 :]
    acorr /= np.arange(len_y, 0, -1)
    acorr /= acorr[..., :1]
    return acorr


def _autocov(x):
    """Compute autocovariance estimates (over the last axis) for every lag for the input array.

    Parameters
    ----------
    x : Numpy array
        An array containing MCMC samples

    Returns
    -------
    acov: Numpy array same size as the input array
    """
    acorr = autocorr(x)
    varx = np.var(x, axis=-1, ddof=0, keepdims=True)
    acov = acorr * varx
    return acov


def rhat(data, var_names=None):
    r"""Compute estimate of Split R-hat for a set of traces.

    The Split R-hat diagnostic tests for lack of convergence by comparing the variance between
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
        For ndarray: shape = (chain, draw).
    var_names : list
      Names of variables to include in the rhat report

    Returns
    -------
    r_hat : xarray.Dataset
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
    Gelman et al. BDA (2014)
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)
    """
    if isinstance(data, np.ndarray):
        return _get_split_rhat(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    return xr.apply_ufunc(_rhat_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def _rhat_ufunc(ary):
    """Ufunc for computing effective sample size.

    This can be used on an xarray Dataset, using
    `xr.apply_ufunc(_neff_ufunc, ..., input_core_dims=(('chain', 'draw'),))
    """
    target = np.empty(ary.shape[:-2])
    for idx in np.ndindex(target.shape):
        target[idx] = _get_split_rhat(ary[idx])
    return target


def _get_split_rhat(values, round_to=2):
    """Compute the split-rhat for a 2d array."""
    shape = values.shape
    if len(shape) != 2:
        raise TypeError("Effective sample size calculation requires 2 dimensional arrays.")
    _, num_samples = shape
    num_split = num_samples // 2
    # Calculate split chain mean
    split_chain_mean1 = np.mean(values[:, :num_split], axis=1)
    split_chain_mean2 = np.mean(values[:, num_split:], axis=1)
    split_chain_mean = np.concatenate((split_chain_mean1, split_chain_mean2))
    # Calculate split chain variance
    split_chain_var1 = np.var(values[:, :num_split], axis=1, ddof=1)
    split_chain_var2 = np.var(values[:, num_split:], axis=1, ddof=1)
    split_chain_var = np.concatenate((split_chain_var1, split_chain_var2))
    # Calculate between-chain variance
    between_chain_variance = num_samples / 2 * np.var(split_chain_mean, ddof=1)
    # Calculate within-chain variance
    within_chain_variance = np.mean(split_chain_var)
    # Estimate of marginal posterior variance
    split_rhat = np.sqrt(
        (between_chain_variance / within_chain_variance + num_samples / 2 - 1) / (num_samples / 2)
    )

    return round(split_rhat, round_to)


def geweke(values, first=0.1, last=0.5, intervals=20):
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
        first_slice = values[start : start + int(first * (end - start))]
        last_slice = values[int(end - last * (end - start)) :]

        z_score = first_slice.mean() - last_slice.mean()
        z_score /= np.sqrt(first_slice.var() + last_slice.var())

        zscores.append([start, z_score])

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
    kcounts, _ = np.histogram(pareto_tail_indices, bins=[-np.Inf, 0.5, 0.7, 1, np.Inf])
    kprop = kcounts / len(pareto_tail_indices) * 100
    df_k = pd.DataFrame(
        dict(_=["(good)", "(ok)", "(bad)", "(very bad)"], Count=kcounts, Pct=kprop)
    ).rename(index={0: "(-Inf, 0.5]", 1: " (0.5, 0.7]", 2: "   (0.7, 1]", 3: "   (1, Inf)"})

    if np.sum(kcounts[1:]) == 0:
        warnings.warn("All Pareto k estimates are good (k < 0.5)")
    elif np.sum(kcounts[2:]) == 0:
        warnings.warn("All Pareto k estimates are ok (k < 0.7)")

    return df_k
