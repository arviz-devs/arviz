"""Diagnostic functions for ArviZ."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

from .stats_utils import (
    make_ufunc as _make_ufunc,
    _autocov,
    _rint,
    _round,
    _quantile,
    check_valid_size as _check_valid_size,
)
from ..data import convert_to_dataset
from ..utils import _var_names


__all__ = [
    "bfmi",
    "effective_sample_size_mean",
    "effective_sample_size_sd",
    "effective_sample_size_bulk",
    "effective_sample_size_tail",
    "effective_sample_size_quantile",
    "rhat",
    "mcse_mean",
    "mcse_sd",
    "mcse_quantile",
    "geweke",
]


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


def effective_sample_size_mean(data, *, var_names=None):
    r"""Calculate estimate of the effective sample size for mean.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
    var_names : list
        Names of variables to include in the effective_sample_size_mean report

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

    Mean estimation calculates estimate for the data.

    .. math:: ess_mean = ess(x)
    .. math:: sq_ess = ess(x**2)
    .. math:: min(ess_mean, sq_ess)

    References
    ----------
    Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008

    https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html Section 15.4.2

    Gelman et al. BDA (2014) Formula 11.8
    """
    if isinstance(data, np.ndarray):
        return _ess_mean(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    ess_mean_ufunc = _make_ufunc(_ess_mean, ravel=False)
    return xr.apply_ufunc(ess_mean_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def effective_sample_size_sd(data, *, var_names=None):
    r"""Calculate estimate of the effective sample size for sd.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
    var_names : list
        Names of variables to include in the effective_sample_size_sd report

    Returns
    -------
    xarray.Dataset
        Return the effective sample size for sd, :math:`\hat{N}_{eff}`

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

    sd estimation calculates estimates both for the data and squared data,
    where ess_sd estimation is the lower one.

    .. math:: ess_mean = ess(x)
    .. math:: sq_ess = ess(x**2)
    .. math:: min(ess_mean, sq_ess)

    References
    ----------
    Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008

    https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html Section 15.4.2

    Gelman et al. BDA (2014) Formula 11.8
    """
    if isinstance(data, np.ndarray):
        return _ess_sd(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    ess_sd_ufunc = _make_ufunc(_ess_sd, ravel=False)
    return xr.apply_ufunc(ess_sd_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def effective_sample_size_bulk(data, *, var_names=None):
    r"""Calculate estimate of the effective sample size for bulk.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
    var_names : list
        Names of variables to include in the effective_sample_size_quantile report

    Returns
    -------
    xarray.Dataset
        Return the effective sample size for bulk, :math:`\hat{N}_{eff}`

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

    tail estimation follows the quantile estimation and is done by computing ess over z-transformed
    ranked boolean vector with quantiles 5% and 95% and by selecting the minimum value.

    .. math:: I05 = x <= quantile(x, 0.05)
    .. math:: I95 = x <= quantile(x, 0.95)
    .. math:: min(ess(I05), ess(95))

    References
    ----------
    Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008

    https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html Section 15.4.2

    Gelman et al. BDA (2014) Formula 11.8
    """
    if isinstance(data, np.ndarray):
        return _ess_bulk(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    ess_bulk_ufunc = _make_ufunc(_ess_bulk, ravel=False)
    return xr.apply_ufunc(ess_bulk_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def effective_sample_size_tail(data, *, var_names=None):
    r"""Calculate estimate of the effective sample size for tail.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
    var_names : list
        Names of variables to include in the effective_sample_size_quantile report

    Returns
    -------
    xarray.Dataset
        Return the effective sample size for tail, :math:`\hat{N}_{eff}`

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

    tail estimation follows the quantile estimation and is done by computing ess over z-transformed
    ranked boolean vector with quantiles 5% and 95% and by selecting the minimum value.

    .. math:: I05 = x <= quantile(x, 0.05)
    .. math:: I95 = x <= quantile(x, 0.05)
    .. math:: min(ess(z-transform(rank(I05))), ess(z-transform(rank(I95))))

    References
    ----------
    Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008

    https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html Section 15.4.2

    Gelman et al. BDA (2014) Formula 11.8
    """
    if isinstance(data, np.ndarray):
        return _ess_tail(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    ess_tail_ufunc = _make_ufunc(_ess_tail, ravel=False)
    return xr.apply_ufunc(ess_tail_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def effective_sample_size_quantile(data, prob, *, var_names=None):
    r"""Calculate estimate of the effective sample size for quantile on specific quantile.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
    prob : float
        quantile at which the ess is calculated.
    var_names : list
        Names of variables to include in the effective_sample_size_quantile report

    Returns
    -------
    xarray.Dataset
        Return the tail effective sample size for specific quantile, :math:`\hat{N}_{eff}`

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

    quantile estimation is done by computing ess over z-transformed and ranked boolean vector

    .. math:: I = x <= quantile(x)
    .. math:: ess(z_transform(rankdata(I)))

    References
    ----------
    Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008

    https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html Section 15.4.2

    Gelman et al. BDA (2014) Formula 11.8
    """
    if isinstance(data, np.ndarray):
        return _ess_quantile(data, prob)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    ess_quantile_ufunc = _make_ufunc(_ess_quantile, ravel=False)
    return xr.apply_ufunc(
        ess_quantile_ufunc, dataset, prob, input_core_dims=(("chain", "draw"), ("chain", "draw"))
    )


def rhat(data, *, var_names=None):
    r"""Compute estimate of rank normalized splitR-hat for a set of traces.

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
      Returns dataset of the potential scale reduction factors, :math:`\hat{R}`

    Notes
    -----
    The diagnostic is computed by:

      .. math:: \hat{R} = \frac{\hat{V}}{W}

    where :math:`W` is the within-chain variance and :math:`\hat{V}` is the posterior variance
    estimate for the pooled rank-traces. This is the potential scale reduction factor, which converges
    to unity when each of the traces is a sample from the target posterior. Values greater than one
    indicate that one or more chains have not yet converged.

    Rank values are calculated over all the chains with `scipy.stats.rankdata`.
    Each chain is split in two and normalized with the z-transform following Vehtari et al. (2019).

    References
    ----------
    Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008
    Gelman et al. BDA (2014)
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)
    """
    if isinstance(data, np.ndarray):
        return _rhat_rank_normalized(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    rhat_ufunc = _make_ufunc(_rhat_rank_normalized, ravel=False)
    return xr.apply_ufunc(rhat_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def mcse_mean(data, *, var_names=None):
    r"""Calculate mcse mean."""
    if isinstance(data, np.ndarray):
        return _mcse_mean(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    mcse_mean_ufunc = _make_ufunc(_mcse_mean, ravel=False)
    return xr.apply_ufunc(mcse_mean_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def mcse_sd(data, *, var_names=None):
    r"""Calculate mcse sd."""
    if isinstance(data, np.ndarray):
        return _mcse_sd(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    mcse_sd_ufunc = _make_ufunc(_mcse_sd, ravel=False)
    return xr.apply_ufunc(mcse_sd_ufunc, dataset, input_core_dims=(("chain", "draw"),))


def mcse_mean_sd(data, *, var_names=None):
    r"""Calculate mcse mean and sd in one go."""
    if isinstance(data, np.ndarray):
        return _mcse_mean_sd(data)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]
    mcse_mean_sd_ufunc = _make_ufunc(_mcse_mean_sd, ravel=False)
    return xr.apply_ufunc(
        mcse_mean_sd_ufunc, dataset, input_core_dims=(("chain", "draw"),), output_core_dims=([], [])
    )


def mcse_quantile(data, prob, *, var_names=None):
    r"""Calculate mcse for quantile."""
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
    _check_valid_size(values, "Rhat")
    _, num_samples = values.shape
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
    return _round(rhat, round_to)


def _split_rhat(values, round_to=2):
    """Compute the split-rhat for a 2d array."""
    _check_valid_size(values, "split-Rhat")
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
    return _round(split_rhat, rount_to)


def _rhat_rank_normalized(ary, round_to=2):
    """Compute the rank normalized rhat for 2d array.

    Computation follows https://arxiv.org/abs/1903.08008
    """
    ary = np.asarray(ary)
    _check_valid_size(ary, "rank normalized split-Rhat")

    rhat_bulk = _rhat(_z_scale(_split_chains(ary)), None)

    ary_folded = np.abs(ary - np.median(ary))
    rhat_tail = _rhat(_z_scale(_split_chains(ary_folded)), None)

    rhat = max(rhat_bulk, rhat_tail)
    return _round(rhat, round_to)


def _ess(sample_array):
    """Compute the effective sample size for a 2D array."""
    sample_array = np.asarray(sample_array)
    shape = sample_array.shape
    _check_valid_size(sample_array, "Effective sample size")
    n_chain, n_draw = shape
    acov = np.asarray([_autocov(sample_array[chain]) for chain in range(n_chain)])
    chain_mean = sample_array.mean(axis=1)
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
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
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

    ess = (
        int((n_chain * n_draw) / (-1.0 + 2.0 * np.sum(rho_hat_t[: max_t + 1])))
        if not np.any(np.isnan(rho_hat_t))
        else np.nan
    )
    return ess


def _ess_bulk(ary):
    """Compute the effective sample size for the bulk."""
    _check_valid_size(ary, "Bulk effective sample size")
    z_split = _z_scale(_split_chains(ary))
    ess_bulk = _ess(z_split)
    return ess_bulk


def _ess_tail(ary):
    """Compute the effective sample size for the tail."""
    _check_valid_size(ary, "Tail effective sample size")
    q05, q95 = _quantile(ary, [0.05, 0.95])
    I05 = ary <= q05
    q05_ess = _ess(_z_scale(_split_chains(I05)))
    I95 = ary <= q95
    q95_ess = _ess(_z_scale(_split_chains(I95)))
    return min(q05_ess, q95_ess)


def _ess_mean(ary):
    """Compute the effective sample size for the mean."""
    _check_valid_size(ary, "Mean effective sample size")
    return _ess(ary)


def _ess_sd(ary):
    """Compute the effective sample size for the sd."""
    _check_valid_size(ary, "SD effective sample size")
    return min(_ess(ary), _ess(ary ** 2))


def _ess_quantile(ary, prob):
    """Compute the effective sample size for the specific resiual."""
    _check_valid_size(ary, "Quantile effective sample size")
    q, = _quantile(ary, prob)
    I = ary <= q
    return _ess(_z_scale(_split_chains(I)))


def _conv_quantile(ary, prob):
    """Return mcse, Q05, Q95, Seff"""
    ess = _ess_quantile(ary, prob)
    p = [0.1586553, 0.8413447, 0.05, 0.95]
    with np.errstate(invalid="ignore"):
        a = stats.beta.ppf(prob, ess * prob + 1, ess * (1 - prob) + 1)
    sorted_ary = np.sort(ary.ravel())
    size = sorted_ary.size
    th1 = sorted_ary[_rint(np.nanmax(a[1] * S, 0))]
    th2 = sorted_ary[_rint(np.nanmin(a[2] * S, size - 1))]
    mcse = (th2 - th1) / 2
    th1 = sorted_ary[_rint(np.nanmax(a[3] * S, 0))]
    th2 = sorted_ary[_rint(np.nanmin(a[4] * S, size - 1))]
    return mcse, th1, th2, ess


def _mcse_mean(ary):
    """Compute the Markov Chain mean error."""
    ess = _ess(ary)
    mean = np.mean(ary)
    sd = np.std(ary, ddof=1)
    mcse_mean = sd / np.sqrt(ess)
    return mcse_mean


def _mcse_sd(ary):
    """Compute the Markov Chain sd error."""
    ess = _ess(ary)
    sd = np.std(ary, ddof=1)

    ess2 = _ess(ary ** 2)
    essmin = min(ess, ess2)
    fac_mcse_sd = np.sqrt(np.exp(1) * (1 - 1 / essmin) ** (essmin - 1) - 1)
    mcse_sd = sd * fac_mcse_sd
    return mcse_sd


def _mcse_mean_sd(ary):
    """Compute the Markov Chain mean and sd errors."""
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
    mcse, *_ = conv_quantile(ary, prob)
    return mcse


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
            - mcse_mean, mcse_sd, ess_mean, ess_sd, ess_bulk, ess_tail, r_hat
    """
    # ess mean
    ess_mean = _ess(ary)

    # ess sd
    ess_sd = _ess_sd(ary)

    # ess bulk
    z_split = _z_scale(_split_chains(ary))
    ess_bulk = _ess(z_split)

    # ess tail
    q05, q95 = _quantile(ary, [0.05, 0.95])
    I05 = ary <= q05
    q05_ess = _ess(_z_scale(_split_chains(I05)))
    I95 = ary <= q95
    q95_ess = _ess(_z_scale(_split_chains(I95)))
    ess_tail = min(q05_ess, q95_ess)

    # r_hat
    rhat_bulk = _rhat(z_split, None)
    ary_folded = np.abs(ary - np.median(ary))
    rhat_tail = _rhat(_z_scale(_split_chains(ary_folded)), None)
    rhat = max(rhat_bulk, rhat_tail)

    # mcse_mean
    mean = np.mean(ary)
    sd = np.std(ary, ddof=1)
    mcse_mean = sd / np.sqrt(ess_mean)

    # mcse_sd
    fac_mcse_sd = np.sqrt(np.exp(1) * (1 - 1 / ess_sd) ** (ess_sd - 1) - 1)
    mcse_sd = sd * fac_mcse_sd

    return mcse_mean, mcse_sd, ess_mean, ess_sd, ess_bulk, ess_tail, rhat
