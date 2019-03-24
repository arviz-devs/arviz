"""Diagnostic functions for ArviZ."""
import warnings

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy import stats

import xarray as xr

from .stats_utils import make_ufunc as _make_ufunc
from ..data import convert_to_dataset
from ..utils import _var_names


__all__ = [
    "bfmi",
    "effective_sample_size",
    "bulk_effective_sample_size",
    "tail_effective_sample_size",
    "rhat",
    "geweke",
    "autocorr",
    "mcse_mean",
    "mcse_sd",
    "mcse_quantile",
]


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
    with warnings.catch_warnings():
        # silence annoying numpy tuple warning in another library
        # silence hack added in 0.3.3+
        warnings.simplefilter("ignore")
        result = fftconvolve(y, y[::-1])
    acorr = result[len(result) // 2 :]
    acorr /= np.arange(len_y, 0, -1)
    with np.errstate(invalid="ignore"):
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


def bfmi(data):
    r"""Calculate the estimated Bayesian fraction of missing information (BFMI).

    BFMI quantifies how well momentum resampling matches the marginal energy distribution. For more
    information on BFMI, see https://arxiv.org/pdf/1604.00695v1.pdf. The current advice is that
    values smaller than 0.3 indicate poor sampling. However, this threshold is provisional and may
    change. See http://mc-stan.org/users/documentation/case-studies/pystan_workflow.html for more
    information.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        If InferenceData, energy variable needs to be found.

    Returns
    -------
    z : array
        The Bayesian fraction of missing information of the model and trace. One element per
        chain in the trace.
    """
    if isinstance(data, np.ndarray):
        return _bfmi(data)

    dataset = convert_to_dataset(data)
    if (not hasattr(dataset, "sample_stats")) or (not hasattr(dataset.sample_stats, "energy")):
        raise TypeError("Energy variable was not found.")
    return _bfmi(dataset.sample_stats.energy)


def effective_sample_size(data, *, var_names=None):
    r"""Calculate estimate of the effective sample size.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
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
        return _ess(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    ess_ufunc = _make_ufunc(_ess, ravel=False)
    return xr.apply_ufunc(ess_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def bulk_effective_sample_size(data, *, var_names=None):
    r"""Calculate estimate of the bulk effective sample size.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
    var_names : list
      Names of variables to include in the effective_sample_size report

    Returns
    -------
    xarray.Dataset
        Return the bulk effective sample size, :math:`\hat{N}_{eff}`

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
        return _bulk_ess(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    bulk_ess_ufunc = _make_ufunc(_bulk_ess, ravel=False)
    return xr.apply_ufunc(bulk_ess_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def tail_effective_sample_size(data, *, var_names=None):
    r"""Calculate estimate of the tail effective sample size.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
    var_names : list
      Names of variables to include in the effective_sample_size report

    Returns
    -------
    xarray.Dataset
        Return the tail effective sample size, :math:`\hat{N}_{eff}`

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
        return _tail_ess(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    tail_ess_ufunc = _make_ufunc(_tail_ess, ravel=False)
    return xr.apply_ufunc(tail_ess_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def rhat(data, var_names=None):
    r"""Compute estimate of rank normalized R-hat for a set of traces.

    The rank normalized R-hat diagnostic tests for lack of convergence by comparing the variance between
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
    Vehtari et al. https://arxiv.org/abs/1903.08008 (2019)
    Gelman et al. BDA (2014)
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)
    """
    if isinstance(data, np.ndarray):
        return _get_split_rhat(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    rhat_ufunc = _make_ufunc(_rhat_rank_normalized, ravel=False)
    return xr.apply_ufunc(_rhat_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def mcse_mean(data, var_names=None):
    r""""""
    if isinstance(data, np.ndarray):
        return _mcse_mean(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    mcse_mean_ufunc = _make_ufunc(_mcse_mean, ravel=False)
    return xr.apply_ufunc(mcse_mean_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def mcse_sd(data, var_names=None):
    r""""""
    if isinstance(data, np.ndarray):
        return _mcse_sd(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    mcse_sd_ufunc = _make_ufunc(_mcse_sd, ravel=False)
    return xr.apply_ufunc(mcse_sd_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def mcse_mean_sd(data, var_names=None):
    r""""""
    if isinstance(data, np.ndarray):
        return _mcse_mean_sd(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    mcse_mean_sd_ufunc = _make_ufunc(_mcse_mean_sd, ravel=False)
    return xr.apply_ufunc(
        mcse_mean_sd_ufunc, dataset, input_core_dims=(("chain", "draw"),), output_core_dims=([], [])
    )


def mcse_quantile(data, prob, var_names=None):
    r""""""
    if isinstance(data, np.ndarray):
        return _mcse_quantile(data, prob)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    mcse_quantile_ufunc = _make_ufunc(_mcse_quantile, ravel=False)
    return xr.apply_ufunc(
        mcse_quantile_ufunc, dataset, prob, input_core_dims=(("chain", "draw"), ("chain", "draw"))
    )


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


def _bfmi(energy):
    r"""Calculate the estimated Bayesian fraction of missing information (BFMI).

    BFMI quantifies how well momentum resampling matches the marginal energy distribution. For more
    information on BFMI, see https://arxiv.org/pdf/1604.00695v1.pdf. The current advice is that
    values smaller than 0.3 indicate poor sampling. However, this threshold is provisional and may
    change. See http://mc-stan.org/users/documentation/case-studies/pystan_workflow.html for more
    information.

    Parameters
    ----------
    energy : NumPy array
        Should be extracted from a gradient based sampler, such as in Stan or PyMC3. Typically,
        after converting a trace or fit to InferenceData, the energy will be in
        `data.sample_stats.energy`.

    Returns
    -------
    z : array
        The Bayesian fraction of missing information of the model and trace. One element per
        chain in the trace.
    """
    energy_mat = np.atleast_2d(energy)
    num = np.square(np.diff(energy_mat, axis=1)).mean(axis=1)  # pylint: disable=no-member
    den = np.var(energy_mat, axis=1)
    return num / den


def _z_scale(ary):
    """Calculate z_scale

    Parameters
    ----------
    ary : np.ndarray

    Returns
    -------
    np.ndarray
    """
    size = ary.size
    rank = stats.rankdata(ary, method="average")
    z = stats.norm.ppf((rank - 0.5) / size)
    z = z.reshape(ary.shape)
    return z


def _split_chains(ary):
    _, n_draw = ary.shape
    half = n_draw // 2
    return np.vstack((ary[:, :half], ary[:, -half:]))


def _rhat(values, round_to=2):
    """Compute the rhat for a 2d array."""
    shape = values.shape
    if len(shape) != 2:
        raise TypeError("Effective sample size calculation requires 2 dimensional arrays.")
    _, num_samples = shape
    # Calculate chain mean
    chain_mean = np.mean(values, axis=1)
    # Calculate chain variance
    chain_var = np.var(values, axis=1, ddof=1)
    # Calculate between-chain variance
    between_chain_variance = num_samples / 2 * np.var(chain_mean, ddof=1)
    # Calculate within-chain variance
    within_chain_variance = np.mean(chain_var)
    # Estimate of marginal posterior variance
    rhat = np.sqrt(
        (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)
    )
    if round_to is None:
        return rhat
    else:
        return round(rhat, round_to)


def _split_rhat(values, round_to=2):
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


def _rhat_rank_normalized(ary, round_to=2):
    # z_scale
    z_split = _z_scale(_split_chains(ary))
    z_split_rhat = _rhat(z_split, None)

    # folded z_scale
    ary_folded = np.abs(ary - np.median(ary))
    z_folded_split = _z_scale(_split_chains(ary_folded))
    z_fsplit_rhat = _rhat(z_folded_split, None)

    rhat = max(z_split_rhat, z_fsplit_rhat)
    if round_to is None:
        return rhat
    else:
        return np.round(rhat, round_to)


def _ess(sample_array):
    """Compute the effective sample size for a 2D array."""
    shape = np.asarray(sample_array).shape
    if len(shape) != 2:
        raise TypeError("Effective sample size calculation requires 2 dimensional arrays.")
    n_chain, n_draws = shape
    if n_chain <= 1:
        raise TypeError("Effective sample size calculation requires multiple chains.")

    acov = np.asarray([_autocov(sample_array[chain]) for chain in range(n_chain)])
    chain_mean = sample_array.mean(axis=1)
    chain_var = acov[:, 0] * n_draws / (n_draws - 1.0)
    acov_t = acov[:, 1] * n_draws / (n_draws - 1.0)
    mean_var = np.mean(chain_var)
    var_plus = mean_var * (n_draws - 1.0) / n_draws
    var_plus += np.var(chain_mean, ddof=1)

    rho_hat_t = np.zeros(n_draws)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - np.mean(acov_t)) / var_plus
    rho_hat_t[1] = rho_hat_odd

    # Geyer's initial positive sequence
    max_t = 1
    t = 1
    while t < (n_draws - 2) and (rho_hat_even + rho_hat_odd) >= 0.0:
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        max_t = t + 2
        t += 2

    # Geyer's initial monotone sequence
    t = 3
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2

    ess = (
        int((n_chain * n_draws) / (-1.0 + 2.0 * np.sum(rho_hat_t)))
        if not np.any(np.isnan(rho_hat_t))
        else np.nan
    )
    return ess


def _bulk_ess(ary):
    z_split = _z_scale(_split_chains(ary))
    bulk_ess = _ess(z_split)
    return bulk_ess


def _tail_ess(ary):
    I05 = ary <= np.quantile(ary, 0.05)
    q05_ess = _ess(_z_scale(_split_chains(I05)))
    I95 = ary <= np.quantile(ary, 0.95)
    q95_ess = _ess(_z_scale(_split_chains(I95)))
    return min(q05_ess, q95_ess)


def _mcse_mean(ary):
    ess = _ess(ary)
    mean = np.mean(ary)
    sd = np.std(ary, ddof=1)
    mcse_mean = sd / np.sqrt(ess)
    return mcse_mean


def _mcse_sd(ary):
    ess = _ess(ary)
    sd = np.std(ary, ddof=1)

    ess2 = _ess(ary ** 2)
    essmin = min(ess, ess2)
    fac_mcse_sd = np.sqrt(np.exp(1) * (1 - 1 / essmin) ** (essmin - 1) - 1)
    mcse_sd = sd * fac_mcse_sd
    return mcse_sd


def _mcse_mean_sd(ary):
    # mean
    ess = _ess(ary)
    mean = np.mean(ary)
    sd = np.std(ary, ddof=1)
    mcse_mean = sd / np.sqrt(ess)

    # sd
    ess2 = _ess(ary ** 2)
    essmin = min(ess, ess2)
    fac_mcse_sd = np.sqrt(np.exp(1) * (1 - 1 / essmin) ** (essmin - 1) - 1)
    mcse_sd = sd * fac_mcse_sd

    return mcse_mean, mcse_sd


def _mcse_quantile(ary, prob):
    """Return mcse, Q05, Q95, Seff"""
    I = ary <= np.quantile(ary, prob)
    size_ess = _ess(_z_scale(_split_chains(I)))
    p = np.array([0.1586553, 0.8413447, 0.05, 0.95])
    with np.errstate(invalid="ignore"):
        a = stats.beta.ppf(p, size_ess * prob + 1, size_ess * (1 - prob) + 1)
    sorted_ary = np.sort(ary.ravel())
    size = ary.size
    th1_idx = int(np.nanmax((round(a[0] * size), 0)))
    th2_idx = int(np.nanmin((round(a[1] * size), size - 1)))
    th1 = sorted_ary[th1_idx]
    th2 = sorted_ary[th2_idx]
    mcse = (th2 - th1) / 2
    th1_idx = int(np.nanmax((round(a[2] * size), 0)))
    th2_idx = int(np.nanmin((round(a[3] * size), size - 1)))
    th1 = sorted_ary[th1_idx]
    th2 = sorted_ary[th2_idx]
    return mcse, th1, th2, size_ess


def _mc_error(x, batches=5, circular=False):
    """Calculate the simulation standard error, accounting for non-independent samples.

    The trace is divided into batches, and the standard deviation of the batch
    means is calculated.

    Parameters
    ----------
    x : Numpy array
        An array containing MCMC samples
    batches : integer
        Number of batches
    circular : bool
        Whether to compute the error taking into account `x` is a circular variable
        (in the range [-np.pi, np.pi]) or not. Defaults to False (i.e non-circular variables).

    Returns
    -------
    mc_error : float
        Simulation standard error
    """
    if x.ndim > 1:

        dims = np.shape(x)
        trace = np.transpose([t.ravel() for t in x])

        return np.reshape([_mc_error(t, batches) for t in trace], dims[1:])

    else:
        if batches == 1:
            if circular:
                std = st.circstd(x, high=np.pi, low=-np.pi)
            else:
                std = np.std(x)
            return std / np.sqrt(len(x))

        batched_traces = np.resize(x, (batches, int(len(x) / batches)))

        if circular:
            means = st.circmean(batched_traces, high=np.pi, low=-np.pi, axis=1)
            std = st.circstd(means, high=np.pi, low=-np.pi)
        else:
            means = np.mean(batched_traces, 1)
            std = np.std(means)

        return std / np.sqrt(batches)


def _multichain_statistics(ary):
    """Calculate efficiently multichain statistics for summary.

    Parameters
    ----------
    ary : numpy.ndarray

    Returns
    -------
    tuple
        Order of return parameters is
            - mcse_mean, mcse_sd, bulk_ess, tail_ess, r_hat
    """
    # Bulk ess
    ary_split = _split_chains(ary)
    z_split = _z_scale(ary_split)
    bulk_ess = _ess(z_split)

    # Tail ess
    I05 = ary <= np.quantile(ary, 0.05)
    q05_ess = _ess(_z_scale(_split_chains(I05)))
    I95 = ary <= np.quantile(ary, 0.95)
    q95_ess = _ess(_z_scale(_split_chains(I95)))
    tail_ess = min(q05_ess, q95_ess)

    # r_hat
    z_split_rhat = _rhat(z_split, None)
    ary_folded = np.abs(ary - np.median(ary))
    z_folded_split = _z_scale(_split_chains(ary_folded))
    z_fsplit_rhat = _rhat(z_folded_split, None)
    rhat = max(z_split_rhat, z_fsplit_rhat)

    # mcse_mean
    ess = _ess(ary)
    mean = np.mean(ary)
    sd = np.std(ary, ddof=1)
    mcse_mean = sd / np.sqrt(ess)

    # mcse_sd
    ess2 = _ess(ary ** 2)
    essmin = min(ess, ess2)
    fac_mcse_sd = np.sqrt(np.exp(1) * (1 - 1 / essmin) ** (essmin - 1) - 1)
    mcse_sd = sd * fac_mcse_sd

    return mcse_mean, mcse_sd, bulk_ess, tail_ess, rhat
