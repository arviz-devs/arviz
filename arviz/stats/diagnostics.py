# pylint: disable=too-many-lines, too-many-function-args
"""Diagnostic functions for ArviZ."""
from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .stats_utils import (
    _autocov,
    _rint,
    _round,
    _quantile,
    check_valid_size as _check_valid_size,
    check_nan as _check_nan,
    wrap_xarray_ufunc as _wrap_xarray_ufunc,
)
from ..data import convert_to_dataset
from ..utils import _var_names


__all__ = ["bfmi", "effective_sample_size", "rhat", "mcse", "geweke"]


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


def effective_sample_size(data, *, var_names=None, method="bulk", relative=False, prob=None):
    r"""Calculate estimate of the effective sample size.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
    var_names : list
        Names of variables to include in the effective_sample_size_mean report
    method : str
        Select ess method. Valid methods are
        - "bulk"
        - "tail"     # prob, optional
        - "quantile" # prob
        - "mean" (old ess)
        - "sd"
        - "median"
        - "split_mad" (mean absolute deviance)
        - "z_scale"
        - "split_folded"
        - "split"
    relative : bool
        Return relative ess
        `ress = ess / N`
    prob : float, optional
        probability value for "tail" and "quantile" ess functions.

    Returns
    -------
    xarray.Dataset
        Return the effective sample size for mean, :math:`\hat{N}_{eff}`

    Notes
    -----
    The basic ess diagnostic is computed by:

    .. math:: \hat{N}_{eff} = \frac{MN}{\hat{\tau}}
    .. math:: \hat{\tau} = -1 + 2 \sum_{t'=0}^K \hat{P}_t'

    where :math:`\hat{\rho}_t` is the estimated _autocorrelation at lag t, and T
    is the first odd positive integer for which the sum
    :math:`\hat{\rho}_{T+1} + \hat{\rho}_{T+1}` is negative.

    The current implementation is similar to Stan, which uses Geyer's initial monotone sequence
    criterion (Geyer, 1992; Geyer, 2011).

    References
    ----------
    Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008
    https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html Section 15.4.2
    Gelman et al. BDA (2014) Formula 11.8
    """
    methods = {
        "bulk": _ess_bulk,
        "tail": _ess_tail,
        "quantile": _ess_quantile,
        "mean": _ess_mean,
        "sd": _ess_sd,
        "median": _ess_median,
        "mad": _ess_mad,
        "z_scale": _ess_z_scale,
        "folded": _ess_folded,
        "split": _ess_split,
    }

    methods_relative = {
        "bulk": _ress_bulk,
        "tail": _ress_tail,
        "quantile": _ress_quantile,
        "mean": _ress_mean,
        "sd": _ress_sd,
        "median": _ress_median,
        "mad": _ress_mad,
        "z_scale": _ress_z_scale,
        "folded": _ress_folded,
        "split": _ress_split,
    }

    if method not in methods:
        raise TypeError(
            "ESS method {} not found. Valid methods are:\n{}".format(method, "\n    ".join(methods))
        )
    if relative:
        ess_func = methods_relative[method]
    else:
        ess_func = methods[method]

    if (method == "quantile") and prob is None:
        raise TypeError("Quantile (prob) information needs to be defined.")

    if isinstance(data, np.ndarray):
        if prob is not None:
            return ess_func(data, prob=prob)
        else:
            return ess_func(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]

    ufunc_kwargs = {"ravel": False}
    func_kwargs = {} if prob is None else {"prob": prob}
    return _wrap_xarray_ufunc(ess_func, dataset, ufunc_kwargs=ufunc_kwargs, func_kwargs=func_kwargs)


def rhat(data, *, var_names=None, method="rank"):
    r"""Compute estimate of rank normalized splitR-hat for a set of traces.

    The rank normalized R-hat diagnostic tests for lack of convergence by comparing the variance
    between multiple chains to the variance within each chain. If convergence has been achieved,
    the between-chain and within-chain variances should be identical. To be most effective in
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
    method : str
        Select R-hat method. Valid methods are
        - "rank"        # recommended by Vehtari et al. (2019)
        - "split"       # old split-Rhat
        - "identity"    # no-split Rhat
        - "folded"
        - "z_scale"

    Returns
    -------
    xarray.Dataset
      Returns dataset of the potential scale reduction factors, :math:`\hat{R}`

    Notes
    -----
    The diagnostic is computed by:

      .. math:: \hat{R} = \frac{\hat{V}}{W}

    where :math:`W` is the within-chain variance and :math:`\hat{V}` is the posterior variance
    estimate for the pooled rank-traces. This is the potential scale reduction factor, which
    converges to unity when each of the traces is a sample from the target posterior. Values
    greater than one indicate that one or more chains have not yet converged.

    Rank values are calculated over all the chains with `scipy.stats.rankdata`.
    Each chain is split in two and normalized with the z-transform following Vehtari et al. (2019).

    References
    ----------
    Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008
    Gelman et al. BDA (2014)
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)
    """
    methods = {
        "rank": _rhat_rank_normalized,
        "split": _rhat_split,
        "identity": _rhat,
        "folded": _rhat_folded,
        "z_scale": _rhat_z_scale,
    }
    if method not in methods:
        raise TypeError(
            "R-hat method {} not found. Valid methods are:\n{}".format(
                method, "\n    ".join(methods)
            )
        )
    rhat_func = methods[method]

    if isinstance(data, np.ndarray):
        return rhat_func(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]

    ufunc_kwargs = {"ravel": False}
    func_kwargs = {}
    return _wrap_xarray_ufunc(
        rhat_func, dataset, ufunc_kwargs=ufunc_kwargs, func_kwargs=func_kwargs
    )


def mcse(data, *, var_names=None, method="mean", prob=None):
    """Calculate Markov Chain Standard Error for statistic.

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
    method : str
        Select mcse method. Valid methods are
        - "mean"
        - "sd"
        - "quantile"
    prob : float
        Quantile information.

    Returns
    -------
    xarray.Dataset
        Return the msce dataset
    """
    methods = {"mean": _mcse_mean, "sd": _mcse_sd, "quantile": _mcse_quantile}
    if method not in methods:
        raise TypeError(
            "mcse method {} not found. Valid methods are:\n{}".format(
                method, "\n    ".join(methods)
            )
        )
    mcse_func = methods[method]

    if method == "quantile" and prob is None:
        raise TypeError("Quantile (prob) information needs to be defined.")

    if isinstance(data, np.ndarray):
        if prob is not None:
            return mcse_func(data, prob=prob)  # pylint: disable=unexpected-keyword-arg
        else:
            return mcse_func(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]

    ufunc_kwargs = {"ravel": False}
    func_kwargs = {} if prob is None else {"prob": prob}
    return _wrap_xarray_ufunc(
        mcse_func, dataset, ufunc_kwargs=ufunc_kwargs, func_kwargs=func_kwargs
    )


def geweke(ary, first=0.1, last=0.5, intervals=20):
    r"""Compute z-scores for convergence diagnostics.

    Compare the mean of the first % of series with the mean of the last % of series. x is divided
    into a number of segments for which this difference is computed. If the series is converged,
    this score should oscillate between -1 and 1.

    Parameters
    ----------
    ary : 1D array-like
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
    end = len(ary) - 1

    # Start intervals going up to the <last>% of the chain
    last_start_idx = (1 - last) * end

    # Calculate starting indices
    start_indices = np.linspace(0, last_start_idx, num=intervals, endpoint=True, dtype=int)

    # Loop over start indices
    for start in start_indices:
        # Calculate slices
        first_slice = ary[start : start + int(first * (end - start))]
        last_slice = ary[int(end - last * (end - start)) :]

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
    """Calculate z_scale.

    Parameters
    ----------
    ary : np.ndarray

    Returns
    -------
    np.ndarray
    """
    ary = np.asarray(ary)
    _check_valid_size(ary, "Bulk effective sample size")
    if _check_nan(ary):
        return np.nan
    size = ary.size
    rank = stats.rankdata(ary, method="average")
    z = stats.norm.ppf((rank - 0.5) / size)
    z = z.reshape(ary.shape)
    return z


def _split_chains(ary):
    """Split and stack chains."""
    ary = np.asarray(ary)
    _check_valid_size(ary, "Bulk effective sample size")
    _, n_draw = ary.shape
    half = n_draw // 2
    return np.vstack((ary[:, :half], ary[:, -half:]))


def _rhat(ary, split=False):
    """Compute the rhat for a 2d array."""
    ary = np.asarray(ary)
    _check_valid_size(ary, "Rhat")
    if _check_nan(ary):
        return np.nan
    _, num_samples = ary.shape
    if split:
        ary = _split_chains(ary)
    # Calculate chain mean
    chain_mean = np.mean(ary, axis=1)
    # Calculate chain variance
    chain_var = np.var(ary, axis=1, ddof=1)
    # Calculate between-chain variance
    between_chain_variance = num_samples / 2 * np.var(chain_mean, ddof=1)
    # Calculate within-chain variance
    within_chain_variance = np.mean(chain_var)
    # Estimate of marginal posterior variance
    rhat_value = np.sqrt(
        (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)
    )
    return rhat_value


def _rhat_split(ary):
    """Compute the split-rhat for a 2d array."""
    return _rhat(ary, split=True)


def _rhat_rank_normalized(ary):
    """Compute the rank normalized rhat for 2d array.

    Computation follows https://arxiv.org/abs/1903.08008
    """
    ary = np.asarray(ary)
    rhat_bulk = _rhat(_z_scale(_split_chains(ary)), None)

    ary_folded = abs(ary - np.median(ary))
    rhat_tail = _rhat(_z_scale(_split_chains(ary_folded)), None)

    rhat_rank = max(rhat_bulk, rhat_tail)
    return rhat_rank


def _rhat_folded(ary):
    """Calculate split-Rhat for folded z-values."""
    ary = _z_split_fold(ary)
    return _rhat(ary)


def _ess(ary, split=False):
    """Compute the effective sample size for a 2D array."""
    ary = np.asarray(ary)
    _check_valid_size(ary, "Effective sample size")
    if _check_nan(ary):
        return np.nan
    if split:
        ary = _split_chains(ary)
    n_chain, n_draw = ary.shape
    acov = np.asarray([_autocov(ary[chain]) for chain in range(n_chain)])
    chain_mean = ary.mean(axis=1)
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += np.var(chain_mean, ddof=1)

    rho_hat_t = np.zeros(n_draw)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho_hat_t[1] = rho_hat_odd

    # Geyer's initial positive sequence
    t = 1
    while t < (n_draw - 2) and (rho_hat_even + rho_hat_odd) >= 0.0:
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        rho_hat_t[t + 1] = rho_hat_even
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2

    max_t = t
    # Geyer's initial monotone sequence
    t = 1
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2

    tau_hat = -1.0 + 2.0 * np.sum(rho_hat_t[:max_t]) + np.sum(rho_hat_t[max_t + 1 : max_t + 2])
    ess = (n_chain * n_draw) / tau_hat
    if np.isnan(rho_hat_t).any():
        ess = np.nan
    return ess


def _ess_bulk(ary):
    """Compute the effective sample size for the bulk."""
    z_split = _z_scale(_split_chains(ary))
    ess_bulk = _ess(z_split)
    return ess_bulk


def _ess_tail(ary, prob=None):
    """Compute the effective sample size for the tail.

    If `prob` defined, ess = min(qess(prob), qess(1-prob))
    """
    if prob is None:
        prob = (0.05, 0.95)
    elif not isinstance(prob, Sequence):
        prob = (prob, 1 - prob)

    prob_low, prob_high = prob
    quantile_low_ess = _ess_quantile(ary, prob_low)
    quantile_high_ess = _ess_quantile(ary, prob_high)
    return min(quantile_low_ess, quantile_high_ess)


def _ess_mean(ary, split=False):
    """Compute the effective sample size for the mean."""
    ary = np.asarray(ary)
    return _ess(ary, split=split)


def _ess_sd(ary, split=False):
    """Compute the effective sample size for the sd."""
    ary = np.asarray(ary)
    return min(_ess(ary, split=split), _ess(ary ** 2, split=split))


def _ess_quantile(ary, prob):
    """Compute the effective sample size for the specific residual."""
    ary = np.asarray(ary)
    _check_valid_size(ary, "Quantile effective sample size")
    if _check_nan(ary):
        return np.nan
    quantile, = _quantile(ary, prob)
    iquantile = ary <= quantile
    return _ess(_z_scale(_split_chains(iquantile)))


def _ess_split(ary):
    """Calculate split-ess."""
    ary = np.asarray(ary)
    _check_valid_size(ary, "Effective sample size")
    return _ess(ary, split=True)


def _ess_z_scale(ary, split=False):
    """Calculate ess for z-scaLe."""
    return _ess(_z_scale(ary), split=split)


def _ess_folded(ary):
    """Calculate split-ess for folded data."""
    return _ess(_z_split_fold(ary))


def _ess_median(ary):
    """Calculate split-ess for median."""
    return _ess_quantile(ary, 0.5)


def _ess_mad(ary):
    """Calculate split-ess for mean absolute deviance."""
    ary = np.asarray(ary)
    _check_valid_size(ary, "Effective sample size")
    if _check_nan(ary):
        return np.nan
    ary = abs(ary - np.median(ary))
    ary = ary <= np.median(ary)
    ary = _z_scale(_split_chains(ary))
    return _ess(ary)


def _conv_quantile(ary, prob):
    """Return mcse, Q05, Q95, Seff."""
    ary = np.asarray(ary)
    if _check_nan(ary):
        return np.nan, np.nan, np.nan, np.nan
    ess = _ess_quantile(ary, prob)
    probability = [0.1586553, 0.8413447, 0.05, 0.95]
    with np.errstate(invalid="ignore"):
        ppf = stats.beta.ppf(probability, ess * prob + 1, ess * (1 - prob) + 1)
    sorted_ary = np.sort(ary.ravel())
    size = sorted_ary.size
    th1 = sorted_ary[_rint(np.nanmax((ppf[0] * size, 0)))]
    th2 = sorted_ary[_rint(np.nanmin((ppf[1] * size, size - 1)))]
    mcse_quantile = (th2 - th1) / 2
    th1 = sorted_ary[_rint(np.nanmax((ppf[2] * size, 0)))]
    th2 = sorted_ary[_rint(np.nanmin((ppf[3] * size, size - 1)))]
    return mcse_quantile, th1, th2, ess


def _mcse_mean(ary, split=False):
    """Compute the Markov Chain mean error."""
    ary = np.asarray(ary)
    if _check_nan(ary):
        return np.nan
    ess = _ess_mean(ary, split=split)
    sd = np.std(ary, ddof=1)
    mcse_mean_value = sd / np.sqrt(ess)
    return mcse_mean_value


def _mcse_sd(ary, split=False):
    """Compute the Markov Chain sd error."""
    ary = np.asarray(ary)
    if _check_nan(ary):
        return np.nan
    ess = _ess_sd(ary, split=split)
    sd = np.std(ary, ddof=1)
    fac_mcse_sd = np.sqrt(np.exp(1) * (1 - 1 / ess) ** (ess - 1) - 1)
    mcse_sd_value = sd * fac_mcse_sd
    return mcse_sd_value


def _mcse_quantile(ary, prob):
    """Compute the Markov Chain quantile error at quantile=prob."""
    ary = np.asarray(ary)
    if _check_nan(ary):
        return np.nan
    mcse_q, *_ = _conv_quantile(ary, prob)
    return mcse_q


def _ress_mean(ary, ess=None, split=False):
    """Relative mean effective sample size."""
    ary = np.asarray(ary)
    if ess is None:
        ess = _ess_mean(ary, split=split)
    return ess / ary.size


def _ress_sd(ary, ess=None, split=False):
    """Relative sd effective sample size."""
    ary = np.asarray(ary)
    if ess is None:
        ess = _ess_sd(ary, split=split)
    return ess / ary.size


def _ress_bulk(ary, ess=None):
    """Relative bulk effective sample size."""
    ary = np.asarray(ary)
    if ess is None:
        ess = _ess_bulk(ary)
    return ess / ary.size


def _ress_tail(ary, prob=None, ess=None):
    """Relative tail effective sample size."""
    ary = np.asarray(ary)
    if ess is None:
        ess = _ess_tail(ary, prob=prob)
    return ess / ary.size


def _ress_quantile(ary, prob=None, ess=None):
    """Relative quantile effective sample size."""
    ary = np.asarray(ary)
    if ess is None:
        if prob is None:
            raise TypeError("Prob needs to be defined if `ess` is None.")
        ess = _ess_quantile(ary, prob=prob)
    return ess / ary.size


def _ress_split(ary, ess=None):
    ary = np.asarray(ary)
    if ess is None:
        ess = _ess_split(ary)
    return ess / ary.size


def _ress_z_scale(ary, ess=None, split=False):
    if ess is None:
        ess = _ess_z_scale(ary, split=split)
    return ess / ary.size


def _rhat_z_scale(ary, split=False):
    return _rhat(_z_scale(ary), split=split)


def _z_split_fold(ary):
    ary = np.asarray(ary)
    ary = abs(ary - np.median(ary))
    ary = _z_scale(_split_chains(ary))
    return ary


def _ress_folded(ary, ess=None):
    if ess is None:
        ess = _ess_folded(ary)
    return ess / ary.size


def _ress_median(ary, ess=None):
    if ess is None:
        ess = _ess_median(ary)
    return ess / ary.size


def _ress_mad(ary, ess=None):
    if ess is None:
        ess = _ess_mad(ary)
    return ess / ary.size


def _mc_error(ary, batches=5, circular=False):
    """Calculate the simulation standard error, accounting for non-independent samples.

    The trace is divided into batches, and the standard deviation of the batch
    means is calculated.

    Parameters
    ----------
    ary : Numpy array
        An array containing MCMC samples
    batches : integer
        Number of batches
    circular : bool
        Whether to compute the error taking into account `ary` is a circular variable
        (in the range [-np.pi, np.pi]) or not. Defaults to False (i.e non-circular variables).

    Returns
    -------
    mc_error : float
        Simulation standard error
    """
    if ary.ndim > 1:

        dims = np.shape(ary)
        trace = np.transpose([t.ravel() for t in ary])

        return np.reshape([_mc_error(t, batches) for t in trace], dims[1:])

    else:
        if _check_nan(ary):
            return np.nan
        if batches == 1:
            if circular:
                std = stats.circstd(ary, high=np.pi, low=-np.pi)
            else:
                std = np.std(ary)
            return std / np.sqrt(len(ary))

        batched_traces = np.resize(ary, (batches, int(len(ary) / batches)))

        if circular:
            means = stats.circmean(batched_traces, high=np.pi, low=-np.pi, axis=1)
            std = stats.circstd(means, high=np.pi, low=-np.pi)
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
            - mcse_mean, mcse_sd, ess_mean, ess_sd, ess_bulk, ess_tail, r_hat
    """
    if _check_nan(ary):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    # ess mean
    ess_mean_value = _ess_mean(ary)

    # ess sd
    ess_sd_value = _ess_sd(ary)

    # ess bulk
    z_split = _z_scale(_split_chains(ary))
    ess_bulk_value = _ess(z_split)

    # ess tail
    quantile05, quantile95 = _quantile(ary, [0.05, 0.95])
    iquantile05 = ary <= quantile05
    quantile05_ess = _ess(_z_scale(_split_chains(iquantile05)))
    iquantile95 = ary <= quantile95
    quantile95_ess = _ess(_z_scale(_split_chains(iquantile95)))
    ess_tail_value = min(quantile05_ess, quantile95_ess)

    # r_hat
    rhat_bulk = _rhat(z_split)
    ary_folded = np.abs(ary - np.median(ary))
    rhat_tail = _rhat(_z_scale(_split_chains(ary_folded)))
    rhat_value = max(rhat_bulk, rhat_tail)

    # mcse_mean
    sd = np.std(ary, ddof=1)
    mcse_mean_value = sd / np.sqrt(ess_mean_value)

    # mcse_sd
    fac_mcse_sd = np.sqrt(np.exp(1) * (1 - 1 / ess_sd_value) ** (ess_sd_value - 1) - 1)
    mcse_sd_value = sd * fac_mcse_sd

    return (
        mcse_mean_value,
        mcse_sd_value,
        ess_mean_value,
        ess_sd_value,
        ess_bulk_value,
        ess_tail_value,
        rhat_value,
    )
