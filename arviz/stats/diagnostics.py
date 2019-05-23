# pylint: disable=too-many-lines, too-many-function-args, redefined-outer-name
"""Diagnostic functions for ArviZ."""
from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .stats_utils import (
    rint as _rint,
    quantile as _quantile,
    autocov as _autocov,
    not_valid as _not_valid,
    wrap_xarray_ufunc as _wrap_xarray_ufunc,
)
from ..data import convert_to_dataset
from ..utils import _var_names


__all__ = ["bfmi", "effective_sample_size", "ess", "rhat", "mcse", "geweke"]


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
        Any object that can be converted to an az.InferenceData object.
        Refer to documentation of az.convert_to_dataset for details.
        If InferenceData, energy variable needs to be found.

    Returns
    -------
    z : array
        The Bayesian fraction of missing information of the model and trace. One element per
        chain in the trace.

    Examples
    --------
    Compute the BFMI of an InferenceData object

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data('radon')
           ...: az.bfmi(data)

    """
    if isinstance(data, np.ndarray):
        return _bfmi(data)

    dataset = convert_to_dataset(data, group="sample_stats")
    if not hasattr(dataset, "energy"):
        raise TypeError("Energy variable was not found.")
    return _bfmi(dataset.energy)


def effective_sample_size(data, *, var_names=None, method="bulk", relative=False, prob=None):
    r"""Calculate estimate of the effective sample size.

    Function deprecated. Use `arviz.ess`.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object.
        Refer to documentation of az.convert_to_dataset for details.
        For ndarray: shape = (chain, draw).
        For n-dimensional ndarray transform first to dataset with az.convert_to_dataset.
    var_names : list
        Names of variables to include in the effective_sample_size_mean report
    method : str
        Select ess method. Valid methods are:

        - "bulk"
        - "tail"     # prob, optional
        - "quantile" # prob
        - "mean" (old ess)
        - "sd"
        - "median"
        - "mad" (mean absolute deviance)
        - "z_scale"
        - "folded"
        - "identity"

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
    * Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008
    * https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html
      Section 15.4.2
    * Gelman et al. BDA (2014) Formula 11.8
    """
    warnings.warn(
        "Function `arviz.effective_sample_size` is deprecated. Use `arviz.ess`", DeprecationWarning
    )
    return ess(data, var_names=var_names, method=method, relative=relative, prob=prob)


def ess(data, *, var_names=None, method="bulk", relative=False, prob=None):
    r"""Calculate estimate of the effective sample size.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object.
        Refer to documentation of az.convert_to_dataset for details.
        For ndarray: shape = (chain, draw).
        For n-dimensional ndarray transform first to dataset with az.convert_to_dataset.
    var_names : list
        Names of variables to include in the effective_sample_size_mean report
    method : str
        Select ess method. Valid methods are:

        - "bulk"
        - "tail"     # prob, optional
        - "quantile" # prob
        - "mean" (old ess)
        - "sd"
        - "median"
        - "mad" (mean absolute deviance)
        - "z_scale"
        - "folded"
        - "identity"

    relative : bool
        Return relative ess
        `ress = ess / n`
    prob : float, optional
        probability value for "tail" and "quantile" ess functions.

    Returns
    -------
    xarray.Dataset
        Return the effective sample size, :math:`\hat{N}_{eff}`

    Notes
    -----
    The basic ess diagnostic is computed by:

    .. math:: \hat{N}_{eff} = \frac{MN}{\hat{\tau}}
    .. math:: \hat{\tau} = -1 + 2 \sum_{t'=0}^K \hat{P}_{t'}

    where :math:`M` is the number of chains, :math:`N` the number of draws,
    :math:`\hat{\rho}_t` is the estimated _autocorrelation at lag :math:`t`, and
    :math:`K` is the last integer for which :math:`\hat{P}_{K} = \hat{\rho}_{2K} +
    \hat{\rho}_{2K+1}` is still positive.

    The current implementation is similar to Stan, which uses Geyer's initial monotone sequence
    criterion (Geyer, 1992; Geyer, 2011).

    References
    ----------
    * Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008
    * https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html
      Section 15.4.2
    * Gelman et al. BDA (2014) Formula 11.8

    Examples
    --------
    Calculate the effective_sample_size using the default arguments:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data('non_centered_eight')
           ...: az.ess(data)

    Calculate the ress of some of the variables

    .. ipython::

        In [1]: az.ess(data, relative=True, var_names=["mu", "theta_t"])

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
        "identity": _ess_identity,
    }

    if method not in methods:
        raise TypeError(
            "ESS method {} not found. Valid methods are:\n{}".format(method, "\n    ".join(methods))
        )
    ess_func = methods[method]

    if (method == "quantile") and prob is None:
        raise TypeError("Quantile (prob) information needs to be defined.")

    if isinstance(data, np.ndarray):
        data = np.atleast_2d(data)
        if len(data.shape) < 3:
            if prob is not None:
                return ess_func(  # pylint: disable=unexpected-keyword-arg
                    data, prob=prob, relative=relative
                )
            else:
                return ess_func(data, relative=relative)
        else:
            msg = (
                "Only uni-dimensional ndarray variables are supported."
                " Please transform first to dataset with `az.convert_to_dataset`."
            )
            raise TypeError(msg)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]

    ufunc_kwargs = {"ravel": False}
    func_kwargs = {"relative": relative} if prob is None else {"prob": prob, "relative": relative}
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
        Any object that can be converted to an az.InferenceData object.
        Refer to documentation of az.convert_to_dataset for details.
        At least 2 posterior chains are needed to compute this diagnostic of one or more
        stochastic parameters.
        For ndarray: shape = (chain, draw).
        For n-dimensional ndarray transform first to dataset with az.convert_to_dataset.
    var_names : list
        Names of variables to include in the rhat report
    method : str
        Select R-hat method. Valid methods are:

        - "rank"        # recommended by Vehtari et al. (2019)
        - "split"
        - "folded"
        - "z_scale"
        - "identity"

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
    * Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008
    * Gelman et al. BDA (2014)
    * Brooks and Gelman (1998)
    * Gelman and Rubin (1992)

    Examples
    --------
    Calculate the R-hat using the default arguments:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("non_centered_eight")
           ...: az.rhat(data)

    Calculate the R-hat of some variables using the folded method:

    .. ipython::

        In [1]: az.rhat(data, var_names=["mu", "theta_t"], method="folded")

    """
    methods = {
        "rank": _rhat_rank,
        "split": _rhat_split,
        "folded": _rhat_folded,
        "z_scale": _rhat_z_scale,
        "identity": _rhat_identity,
    }
    if method not in methods:
        raise TypeError(
            "R-hat method {} not found. Valid methods are:\n{}".format(
                method, "\n    ".join(methods)
            )
        )
    rhat_func = methods[method]

    if isinstance(data, np.ndarray):
        data = np.atleast_2d(data)
        if len(data.shape) < 3:
            return rhat_func(data)
        else:
            msg = (
                "Only uni-dimensional ndarray variables are supported."
                " Please transform first to dataset with `az.convert_to_dataset`."
            )
            raise TypeError(msg)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]

    ufunc_kwargs = {"ravel": False}
    func_kwargs = {}
    return _wrap_xarray_ufunc(
        rhat_func, dataset, ufunc_kwargs=ufunc_kwargs, func_kwargs=func_kwargs
    )


def mcse(data, *, var_names=None, method="mean", prob=None):
    """Calculate Markov Chain Standard Error statistic.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
        For ndarray: shape = (chain, draw).
        For n-dimensional ndarray transform first to dataset with az.convert_to_dataset.
    var_names : list
        Names of variables to include in the rhat report
    method : str
        Select mcse method. Valid methods are:

        - "mean"
        - "sd"
        - "quantile"

    prob : float
        Quantile information.

    Returns
    -------
    xarray.Dataset
        Return the msce dataset

    Examples
    --------
    Calculate the Markov Chain Standard Error using the default arguments:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("non_centered_eight")
           ...: az.mcse(data)

    Calculate the Markov Chain Standard Error using the quantile method:

    .. ipython::

        In [1]: az.mcse(data, method="quantile", prob=.7)

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
        data = np.atleast_2d(data)
        if len(data.shape) < 3:
            if prob is not None:
                return mcse_func(data, prob=prob)  # pylint: disable=unexpected-keyword-arg
            else:
                return mcse_func(data)
        else:
            msg = (
                "Only uni-dimensional ndarray variables are supported."
                " Please transform first to dataset with `az.convert_to_dataset`."
            )
            raise TypeError(msg)

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
    * Geweke (1992)
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
    size = ary.size
    rank = stats.rankdata(ary, method="average")
    z = stats.norm.ppf((rank - 0.5) / size)
    z = z.reshape(ary.shape)
    return z


def _split_chains(ary):
    """Split and stack chains."""
    ary = np.asarray(ary)
    if len(ary.shape) > 1:
        _, n_draw = ary.shape
    else:
        ary = np.atleast_2d(ary)
        _, n_draw = ary.shape
    half = n_draw // 2
    return np.vstack((ary[:, :half], ary[:, -half:]))


def _z_fold(ary):
    """Fold and z-scale values."""
    ary = np.asarray(ary)
    ary = abs(ary - np.median(ary))
    ary = _z_scale(ary)
    return ary


def _rhat(ary):
    """Compute the rhat for a 2d array."""
    ary = np.asarray(ary, dtype=float)
    if _not_valid(ary, check_shape=False):
        return np.nan
    _, num_samples = ary.shape

    # Calculate chain mean
    chain_mean = np.mean(ary, axis=1)
    # Calculate chain variance
    chain_var = np.var(ary, axis=1, ddof=1)
    # Calculate between-chain variance
    between_chain_variance = num_samples * np.var(chain_mean, ddof=1)
    # Calculate within-chain variance
    within_chain_variance = np.mean(chain_var)
    # Estimate of marginal posterior variance
    rhat_value = np.sqrt(
        (between_chain_variance / within_chain_variance + num_samples - 1) / (num_samples)
    )
    return rhat_value


def _rhat_rank(ary):
    """Compute the rank normalized rhat for 2d array.

    Computation follows https://arxiv.org/abs/1903.08008
    """
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=2)):
        return np.nan
    split_ary = _split_chains(ary)
    rhat_bulk = _rhat(_z_scale(split_ary))

    split_ary_folded = abs(split_ary - np.median(split_ary))
    rhat_tail = _rhat(_z_scale(split_ary_folded))

    rhat_rank = max(rhat_bulk, rhat_tail)
    return rhat_rank


def _rhat_folded(ary):
    """Calculate split-Rhat for folded z-values."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=2)):
        return np.nan
    ary = _z_fold(_split_chains(ary))
    return _rhat(ary)


def _rhat_z_scale(ary):
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=2)):
        return np.nan
    return _rhat(_z_scale(_split_chains(ary)))


def _rhat_split(ary):
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=2)):
        return np.nan
    return _rhat(_split_chains(ary))


def _rhat_identity(ary):
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=2)):
        return np.nan
    return _rhat(ary)


def _ess(ary, relative=False):
    """Compute the effective sample size for a 2D array."""
    ary = np.asarray(ary, dtype=float)
    if _not_valid(ary, check_shape=False):
        return np.nan
    if (np.max(ary) - np.min(ary)) < np.finfo(float).resolution:  # pylint: disable=no-member
        return ary.size
    if len(ary.shape) < 2:
        ary = np.atleast_2d(ary)
    n_chain, n_draw = ary.shape
    acov = _autocov(ary, axis=1)
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
    while t < (n_draw - 3) and (rho_hat_even + rho_hat_odd) > 0.0:
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2

    max_t = t - 2
    # improve estimation
    if rho_hat_even > 0:
        rho_hat_t[max_t + 1] = rho_hat_even
    # Geyer's initial monotone sequence
    t = 1
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2

    ess = n_chain * n_draw
    tau_hat = -1.0 + 2.0 * np.sum(rho_hat_t[: max_t + 1]) + np.sum(rho_hat_t[max_t + 1 : max_t + 2])
    tau_hat = max(tau_hat, 1 / np.log10(ess))
    ess = (1 if relative else ess) / tau_hat
    if np.isnan(rho_hat_t).any():
        ess = np.nan
    return ess


def _ess_bulk(ary, relative=False):
    """Compute the effective sample size for the bulk."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    z_scaled = _z_scale(_split_chains(ary))
    ess_bulk = _ess(z_scaled, relative=relative)
    return ess_bulk


def _ess_tail(ary, prob=None, relative=False):
    """Compute the effective sample size for the tail.

    If `prob` defined, ess = min(qess(prob), qess(1-prob))
    """
    if prob is None:
        prob = (0.05, 0.95)
    elif not isinstance(prob, Sequence):
        prob = (prob, 1 - prob)

    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan

    prob_low, prob_high = prob
    quantile_low_ess = _ess_quantile(ary, prob_low, relative=relative)
    quantile_high_ess = _ess_quantile(ary, prob_high, relative=relative)
    return min(quantile_low_ess, quantile_high_ess)


def _ess_mean(ary, relative=False):
    """Compute the effective sample size for the mean."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    return _ess(_split_chains(ary), relative=relative)


def _ess_sd(ary, relative=False):
    """Compute the effective sample size for the sd."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    ary = _split_chains(ary)
    return min(_ess(ary, relative=relative), _ess(ary ** 2, relative=relative))


def _ess_quantile(ary, prob, relative=False):
    """Compute the effective sample size for the specific residual."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    if prob is None:
        raise TypeError("Prob not defined.")
    quantile, = _quantile(ary, prob)
    iquantile = ary <= quantile
    return _ess(_split_chains(iquantile), relative=relative)


def _ess_z_scale(ary, relative=False):
    """Calculate ess for z-scaLe."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    return _ess(_z_scale(_split_chains(ary)), relative=relative)


def _ess_folded(ary, relative=False):
    """Calculate split-ess for folded data."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    return _ess(_z_fold(_split_chains(ary)), relative=relative)


def _ess_median(ary, relative=False):
    """Calculate split-ess for median."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    return _ess_quantile(ary, 0.5, relative=relative)


def _ess_mad(ary, relative=False):
    """Calculate split-ess for mean absolute deviance."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    ary = abs(ary - np.median(ary))
    ary = ary <= np.median(ary)
    ary = _z_scale(_split_chains(ary))
    return _ess(ary, relative=relative)


def _ess_identity(ary, relative=False):
    """Calculate ess."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    return _ess(ary, relative=relative)


def _conv_quantile(ary, prob):
    """Return mcse, Q05, Q95, Seff."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan, np.nan, np.nan, np.nan
    ess = _ess_quantile(ary, prob)
    probability = [0.1586553, 0.8413447, 0.05, 0.95]
    with np.errstate(invalid="ignore"):
        ppf = stats.beta.ppf(probability, ess * prob + 1, ess * (1 - prob) + 1)
    sorted_ary = np.sort(ary.ravel())
    size = sorted_ary.size
    ppf_size = ppf * size - 1
    th1 = sorted_ary[_rint(np.nanmax((ppf_size[0], 0)))]
    th2 = sorted_ary[_rint(np.nanmin((ppf_size[1], size - 1)))]
    mcse_quantile = (th2 - th1) / 2
    th1 = sorted_ary[_rint(np.nanmax((ppf_size[2], 0)))]
    th2 = sorted_ary[_rint(np.nanmin((ppf_size[3], size - 1)))]
    return mcse_quantile, th1, th2, ess


def _mcse_mean(ary):
    """Compute the Markov Chain mean error."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    ess = _ess_mean(ary)
    sd = np.std(ary, ddof=1)
    mcse_mean_value = sd / np.sqrt(ess)
    return mcse_mean_value


def _mcse_sd(ary):
    """Compute the Markov Chain sd error."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    ess = _ess_sd(ary)
    sd = np.std(ary, ddof=1)
    fac_mcse_sd = np.sqrt(np.exp(1) * (1 - 1 / ess) ** (ess - 1) - 1)
    mcse_sd_value = sd * fac_mcse_sd
    return mcse_sd_value


def _mcse_quantile(ary, prob):
    """Compute the Markov Chain quantile error at quantile=prob."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    mcse_q, *_ = _conv_quantile(ary, prob)
    return mcse_q


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
        if _not_valid(ary, check_shape=False):
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
    ary = np.atleast_2d(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
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
    quantile05_ess = _ess(_split_chains(iquantile05))
    iquantile95 = ary <= quantile95
    quantile95_ess = _ess(_split_chains(iquantile95))
    ess_tail_value = min(quantile05_ess, quantile95_ess)

    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=2)):
        rhat_value = np.nan
    else:
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
