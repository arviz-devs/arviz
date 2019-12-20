# pylint: disable=too-many-lines
"""Statistical functions in ArviZ."""
import warnings
import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Optional, List, Union

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize
import xarray as xr

from ..data import convert_to_inference_data, convert_to_dataset, InferenceData, CoordSpec, DimSpec
from ..plots.kdeplot import _fast_kde
from ..plots.plot_utils import get_bins
from .diagnostics import _multichain_statistics, _mc_error, ess, _circular_standard_deviation
from .stats_utils import (
    make_ufunc as _make_ufunc,
    wrap_xarray_ufunc as _wrap_xarray_ufunc,
    logsumexp as _logsumexp,
    ELPDData,
    stats_variance_2d as svar,
)
from ..utils import _var_names, Numba, _numba_var
from ..stats.stats_utils import histogram
from ..rcparams import rcParams

_log = logging.getLogger(__name__)

__all__ = [
    "apply_test_function",
    "compare",
    "hpd",
    "loo",
    "loo_pit",
    "psislw",
    "r2_score",
    "summary",
    "waic",
]


def compare(
    dataset_dict,
    ic=None,
    method="BB-pseudo-BMA",
    b_samples=1000,
    alpha=1,
    seed=None,
    scale="deviance",
):
    r"""Compare models based on WAIC or LOO cross-validation.

    WAIC is the widely applicable information criterion, and LOO is leave-one-out
    (LOO) cross-validation. Read more theory here - in a paper by some of the
    leading authorities on model selection - dx.doi.org/10.1111/1467-9868.00353

    Parameters
    ----------
    dataset_dict : dict[str] -> InferenceData
        A dictionary of model names and InferenceData objects
    ic : str
        Information Criterion (WAIC or LOO) used to compare models. Defaults to
        ``rcParams["stats.information_criterion"]``.
    method : str
        Method used to estimate the weights for each model. Available options are:

        - 'stacking' : stacking of predictive distributions.
        - 'BB-pseudo-BMA' : (default) pseudo-Bayesian Model averaging using Akaike-type
          weighting. The weights are stabilized using the Bayesian bootstrap.
        - 'pseudo-BMA': pseudo-Bayesian Model averaging using Akaike-type
          weighting, without Bootstrap stabilization (not recommended).

        For more information read https://arxiv.org/abs/1704.02030
    b_samples: int
        Number of samples taken by the Bayesian bootstrap estimation.
        Only useful when method = 'BB-pseudo-BMA'.
    alpha : float
        The shape parameter in the Dirichlet distribution used for the Bayesian bootstrap. Only
        useful when method = 'BB-pseudo-BMA'. When alpha=1 (default), the distribution is uniform
        on the simplex. A smaller alpha will keeps the final weights more away from 0 and 1.
    seed : int or np.random.RandomState instance
        If int or RandomState, use it for seeding Bayesian bootstrap. Only
        useful when method = 'BB-pseudo-BMA'. Default None the global
        np.random state is used.
    scale : str
        Output scale for IC. Available options are:

        - `deviance` : (default) -2 * (log-score)
        - `log` : 1 * log-score (after Vehtari et al. (2017))
        - `negative_log` : -1 * (log-score)

    Returns
    -------
    A DataFrame, ordered from best to worst model (measured by information criteria).
    The index reflects the key with which the models are passed to this function. The columns are:
    rank : The rank-order of the models. 0 is the best.
    IC : Information Criteria (WAIC or LOO).
        Smaller IC indicates higher out-of-sample predictive fit ("better" model). Default WAIC.
        If `scale == log` higher IC indicates higher out-of-sample predictive fit ("better" model).
    pIC : Estimated effective number of parameters.
    dIC : Relative difference between each IC (WAIC or LOO) and the lowest IC (WAIC or LOO).
        It's always 0 for the top-ranked model.
    weight: Relative weight for each model.
        This can be loosely interpreted as the probability of each model (among the compared model)
        given the data. By default the uncertainty in the weights estimation is considered using
        Bayesian bootstrap.
    SE : Standard error of the IC estimate.
        If method = BB-pseudo-BMA these values are estimated using Bayesian bootstrap.
    dSE : Standard error of the difference in IC between each model and the top-ranked model.
        It's always 0 for the top-ranked model.
    warning : A value of 1 indicates that the computation of the IC may not be reliable.
        This could be indication of WAIC/LOO starting to fail see
        http://arxiv.org/abs/1507.04544 for details.
    scale : Scale used for the IC.

    Examples
    --------
    Compare the centered and non centered models of the eight school problem:

    .. ipython::

        In [1]: import arviz as az
           ...: data1 = az.load_arviz_data("non_centered_eight")
           ...: data2 = az.load_arviz_data("centered_eight")
           ...: compare_dict = {"non centered": data1, "centered": data2}
           ...: az.compare(compare_dict)

    Compare the models using LOO-CV, returning the IC in log scale and calculating the
    weights using the stacking method.

    .. ipython::

        In [1]: az.compare(compare_dict, ic="loo", method="stacking", scale="log")

    """
    names = list(dataset_dict.keys())
    scale = scale.lower()
    if scale == "log":
        scale_value = 1
        ascending = False
    else:
        if scale == "negative_log":
            scale_value = -1
        else:
            scale_value = -2
        ascending = True

    ic = rcParams["stats.information_criterion"] if ic is None else ic.lower()
    if ic == "waic":
        ic_func = waic
        df_comp = pd.DataFrame(
            index=names,
            columns=[
                "rank",
                "waic",
                "p_waic",
                "d_waic",
                "weight",
                "se",
                "dse",
                "warning",
                "waic_scale",
            ],
        )
        scale_col = "waic_scale"

    elif ic == "loo":
        ic_func = loo
        df_comp = pd.DataFrame(
            index=names,
            columns=[
                "rank",
                "loo",
                "p_loo",
                "d_loo",
                "weight",
                "se",
                "dse",
                "warning",
                "loo_scale",
            ],
        )
        scale_col = "loo_scale"

    else:
        raise NotImplementedError("The information criterion {} is not supported.".format(ic))

    if method.lower() not in ["stacking", "bb-pseudo-bma", "pseudo-bma"]:
        raise ValueError("The method {}, to compute weights, is not supported.".format(method))

    ic_se = "{}_se".format(ic)
    p_ic = "p_{}".format(ic)
    ic_i = "{}_i".format(ic)

    ics = pd.DataFrame()
    names = []
    for name, dataset in dataset_dict.items():
        names.append(name)
        ics = ics.append([ic_func(dataset, pointwise=True, scale=scale)])
    ics.index = names
    ics.sort_values(by=ic, inplace=True, ascending=ascending)
    ics[ic_i] = ics[ic_i].apply(lambda x: x.values.flatten())

    if method.lower() == "stacking":
        rows, cols, ic_i_val = _ic_matrix(ics, ic_i)
        exp_ic_i = np.exp(ic_i_val / scale_value)
        last_col = cols - 1

        def w_fuller(weights):
            return np.concatenate((weights, [max(1.0 - np.sum(weights), 0.0)]))

        def log_score(weights):
            w_full = w_fuller(weights)
            score = 0.0
            for i in range(rows):
                score += np.log(np.dot(exp_ic_i[i], w_full))
            return -score

        def gradient(weights):
            w_full = w_fuller(weights)
            grad = np.zeros(last_col)
            for k in range(last_col - 1):
                for i in range(rows):
                    grad[k] += (exp_ic_i[i, k] - exp_ic_i[i, last_col]) / np.dot(
                        exp_ic_i[i], w_full
                    )
            return -grad

        theta = np.full(last_col, 1.0 / cols)
        bounds = [(0.0, 1.0) for _ in range(last_col)]
        constraints = [
            {"type": "ineq", "fun": lambda x: 1.0 - np.sum(x)},
            {"type": "ineq", "fun": np.sum},
        ]

        weights = minimize(
            fun=log_score, x0=theta, jac=gradient, bounds=bounds, constraints=constraints
        )

        weights = w_fuller(weights["x"])
        ses = ics[ic_se]

    elif method.lower() == "bb-pseudo-bma":
        rows, cols, ic_i_val = _ic_matrix(ics, ic_i)
        ic_i_val = ic_i_val * rows

        b_weighting = st.dirichlet.rvs(alpha=[alpha] * rows, size=b_samples, random_state=seed)
        weights = np.zeros((b_samples, cols))
        z_bs = np.zeros_like(weights)
        for i in range(b_samples):
            z_b = np.dot(b_weighting[i], ic_i_val)
            u_weights = np.exp((z_b - np.min(z_b)) / scale_value)
            z_bs[i] = z_b  # pylint: disable=unsupported-assignment-operation
            weights[i] = u_weights / np.sum(u_weights)

        weights = weights.mean(axis=0)
        ses = pd.Series(z_bs.std(axis=0), index=names)  # pylint: disable=no-member

    elif method.lower() == "pseudo-bma":
        min_ic = ics.iloc[0][ic]
        z_rv = np.exp((ics[ic] - min_ic) / scale_value)
        weights = z_rv / np.sum(z_rv)
        ses = ics[ic_se]

    if np.any(weights):
        min_ic_i_val = ics[ic_i].iloc[0]
        for idx, val in enumerate(ics.index):
            res = ics.loc[val]
            if scale_value < 0:
                diff = res[ic_i] - min_ic_i_val
            else:
                diff = min_ic_i_val - res[ic_i]
            d_ic = np.sum(diff)
            d_std_err = np.sqrt(len(diff) * np.var(diff))
            std_err = ses.loc[val]
            weight = weights[idx]
            df_comp.at[val] = (
                idx,
                res[ic],
                res[p_ic],
                d_ic,
                weight,
                std_err,
                d_std_err,
                res["warning"],
                res[scale_col],
            )

    return df_comp.sort_values(by=ic, ascending=ascending)


def _ic_matrix(ics, ic_i):
    """Store the previously computed pointwise predictive accuracy values (ics) in a 2D matrix."""
    cols, _ = ics.shape
    rows = len(ics[ic_i].iloc[0])
    ic_i_val = np.zeros((rows, cols))

    for idx, val in enumerate(ics.index):
        ic = ics.loc[val][ic_i]

        if len(ic) != rows:
            raise ValueError("The number of observations should be the same across all models")

        ic_i_val[:, idx] = ic

    return rows, cols, ic_i_val


def hpd(ary, credible_interval=0.94, circular=False, multimodal=False):
    """
    Calculate highest posterior density (HPD) of array for given credible_interval.

    The HPD is the minimum width Bayesian credible interval (BCI).

    Parameters
    ----------
    ary : Numpy array
        An array containing posterior samples
    credible_interval : float, optional
        Credible interval to compute. Defaults to 0.94.
    circular : bool, optional
        Whether to compute the hpd taking into account `x` is a circular variable
        (in the range [-np.pi, np.pi]) or not. Defaults to False (i.e non-circular variables).
        Only works if multimodal is False.
    multimodal : bool
        If true it may compute more than one hpd interval if the distribution is multimodal and the
        modes are well separated.

    Returns
    -------
    np.ndarray
        lower(s) and upper(s) values of the interval(s).

    Examples
    --------
    Calculate the hpd of a Normal random variable:

    .. ipython::

        In [1]: import arviz as az
           ...: import numpy as np
           ...: data = np.random.normal(size=2000)
           ...: az.hpd(data, credible_interval=.68)
    """
    if ary.ndim > 1:
        hpd_array = np.array(
            [
                hpd(
                    row,
                    credible_interval=credible_interval,
                    circular=circular,
                    multimodal=multimodal,
                )
                for row in ary.T
            ]
        )
        return hpd_array

    if multimodal:
        if ary.dtype.kind == "f":
            density, lower, upper = _fast_kde(ary)
            range_x = upper - lower
            dx = range_x / len(density)
            bins = np.linspace(lower, upper, len(density))
        else:
            bins = get_bins(ary)
            _, density, _ = histogram(ary, bins=bins)
            dx = np.diff(bins)[0]

        density *= dx

        idx = np.argsort(-density)
        intervals = bins[idx][density[idx].cumsum() <= credible_interval]
        intervals.sort()

        intervals_splitted = np.split(intervals, np.where(np.diff(intervals) >= dx * 1.1)[0] + 1)

        hpd_intervals = []
        for interval in intervals_splitted:
            if interval.size == 0:
                hpd_intervals.append((bins[0], bins[0]))
            else:
                hpd_intervals.append((interval[0], interval[-1]))

        hpd_intervals = np.array(hpd_intervals)

    else:
        ary = ary.copy()
        n = len(ary)

        if circular:
            mean = st.circmean(ary, high=np.pi, low=-np.pi)
            ary = ary - mean
            ary = np.arctan2(np.sin(ary), np.cos(ary))

        ary = np.sort(ary)
        interval_idx_inc = int(np.floor(credible_interval * n))
        n_intervals = n - interval_idx_inc
        interval_width = ary[interval_idx_inc:] - ary[:n_intervals]

        if len(interval_width) == 0:
            raise ValueError(
                "Too few elements for interval calculation. "
                "Check that credible_interval meets condition 0 =< credible_interval < 1"
            )

        min_idx = np.argmin(interval_width)
        hdi_min = ary[min_idx]
        hdi_max = ary[min_idx + interval_idx_inc]

        if circular:
            hdi_min = hdi_min + mean
            hdi_max = hdi_max + mean
            hdi_min = np.arctan2(np.sin(hdi_min), np.cos(hdi_min))
            hdi_max = np.arctan2(np.sin(hdi_max), np.cos(hdi_max))

        hpd_intervals = np.array([hdi_min, hdi_max])

    return hpd_intervals


def loo(data, pointwise=False, reff=None, scale="deviance"):
    """Pareto-smoothed importance sampling leave-one-out cross-validation.

    Calculates leave-one-out (LOO) cross-validation for out of sample predictive model fit,
    following Vehtari et al. (2017). Cross-validation is computed using Pareto-smoothed
    importance sampling (PSIS).

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object. Refer to documentation
        of az.convert_to_inference_data for details
    pointwise : bool, optional
        if True the pointwise predictive accuracy will be returned. Defaults to False
    reff : float, optional
        Relative MCMC efficiency, `ess / n` i.e. number of effective samples divided by
        the number of actual samples. Computed from trace by default.
    scale : str
        Output scale for loo. Available options are:

        - `deviance` : (default) -2 * (log-score)
        - `log` : 1 * log-score (after Vehtari et al. (2017))
        - `negative_log` : -1 * (log-score)

        A higher log-score (or a lower deviance) indicates a model with better predictive
        accuracy.

    Returns
    -------
    pandas.Series with the following rows:
    loo : approximated Leave-one-out cross-validation
    loo_se : standard error of loo
    p_loo : effective number of parameters
    shape_warn : bool
        True if the estimated shape parameter of
        Pareto distribution is greater than 0.7 for one or more samples
    loo_i : array of pointwise predictive accuracy, only if pointwise True
    pareto_k : array of Pareto shape values, only if pointwise True
    loo_scale : scale of the loo results

        The returned object has a custom print method that overrides pd.Series method. It is
        specific to expected log pointwise predictive density (elpd) information criteria.

    Examples
    --------
    Calculate the LOO-CV of a model:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("centered_eight")
           ...: az.loo(data)

    The custom print method can be seen here, printing only the relevant information and
    with a specific organization. ``IC_loo`` stands for information criteria, which is the
    `deviance` scale, the `log` (and `negative_log`) correspond to ``elpd`` (and ``-elpd``)

    .. ipython::

        In [2]: az.loo(data, pointwise=True, scale="log")

    """
    inference_data = convert_to_inference_data(data)
    if not hasattr(inference_data, "sample_stats"):
        raise TypeError("Must be able to extract a sample_stats group from data.")
    if "log_likelihood" not in inference_data.sample_stats:
        raise TypeError("Data must include log_likelihood in sample_stats")
    log_likelihood = inference_data.sample_stats.log_likelihood
    log_likelihood = log_likelihood.stack(sample=("chain", "draw"))
    shape = log_likelihood.shape
    n_samples = shape[-1]
    n_data_points = np.product(shape[:-1])

    if scale.lower() == "deviance":
        scale_value = -2
    elif scale.lower() == "log":
        scale_value = 1
    elif scale.lower() == "negative_log":
        scale_value = -1
    else:
        raise TypeError('Valid scale values are "deviance", "log", "negative_log"')

    if reff is None:
        if not hasattr(inference_data, "posterior"):
            raise TypeError("Must be able to extract a posterior group from data.")
        posterior = inference_data.posterior
        n_chains = len(posterior.chain)
        if n_chains == 1:
            reff = 1.0
        else:
            ess_p = ess(posterior, method="mean")
            # this mean is over all data variables
            reff = (
                np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples
            )

    log_weights, pareto_shape = psislw(-log_likelihood, reff)
    log_weights += log_likelihood

    warn_mg = False
    if np.any(pareto_shape > 0.7):
        warnings.warn(
            "Estimated shape parameter of Pareto distribution is greater than 0.7 for "
            "one or more samples. You should consider using a more robust model, this is because "
            "importance sampling is less likely to work well if the marginal posterior and "
            "LOO posterior are very different. This is more likely to happen with a non-robust "
            "model and highly influential observations."
        )
        warn_mg = True

    ufunc_kwargs = {"n_dims": 1, "ravel": False}
    kwargs = {"input_core_dims": [["sample"]]}
    loo_lppd_i = scale_value * _wrap_xarray_ufunc(
        _logsumexp, log_weights, ufunc_kwargs=ufunc_kwargs, **kwargs
    )
    loo_lppd = loo_lppd_i.values.sum()
    loo_lppd_se = (n_data_points * np.var(loo_lppd_i.values)) ** 0.5

    lppd = np.sum(
        _wrap_xarray_ufunc(
            _logsumexp,
            log_likelihood,
            func_kwargs={"b_inv": n_samples},
            ufunc_kwargs=ufunc_kwargs,
            **kwargs,
        ).values
    )
    p_loo = lppd - loo_lppd / scale_value

    if pointwise:
        if np.equal(loo_lppd, loo_lppd_i).all():  # pylint: disable=no-member
            warnings.warn(
                "The point-wise LOO is the same with the sum LOO, please double check "
                "the Observed RV in your model to make sure it returns element-wise logp."
            )
        return ELPDData(
            data=[
                loo_lppd,
                loo_lppd_se,
                p_loo,
                n_samples,
                n_data_points,
                warn_mg,
                loo_lppd_i.rename("loo_i"),
                pareto_shape,
                scale,
            ],
            index=[
                "loo",
                "loo_se",
                "p_loo",
                "n_samples",
                "n_data_points",
                "warning",
                "loo_i",
                "pareto_k",
                "loo_scale",
            ],
        )

    else:
        return ELPDData(
            data=[loo_lppd, loo_lppd_se, p_loo, n_samples, n_data_points, warn_mg, scale],
            index=["loo", "loo_se", "p_loo", "n_samples", "n_data_points", "warning", "loo_scale"],
        )


def psislw(log_weights, reff=1.0):
    """
    Pareto smoothed importance sampling (PSIS).

    Parameters
    ----------
    log_weights : array
        Array of size (n_observations, n_samples)
    reff : float
        relative MCMC efficiency, `ess / n`

    Returns
    -------
    lw_out : array
        Smoothed log weights
    kss : array
        Pareto tail indices
    """
    if hasattr(log_weights, "sample"):
        n_samples = len(log_weights.sample)
        shape = [size for size, dim in zip(log_weights.shape, log_weights.dims) if dim != "sample"]
    else:
        n_samples = log_weights.shape[-1]
        shape = log_weights.shape[:-1]
    # precalculate constants
    cutoff_ind = -int(np.ceil(min(n_samples / 5.0, 3 * (n_samples / reff) ** 0.5))) - 1
    cutoffmin = np.log(np.finfo(float).tiny)  # pylint: disable=no-member, assignment-from-no-return
    k_min = 1.0 / 3

    # create output array with proper dimensions
    out = tuple([np.empty_like(log_weights), np.empty(shape)])

    # define kwargs
    func_kwargs = {"cutoff_ind": cutoff_ind, "cutoffmin": cutoffmin, "k_min": k_min, "out": out}
    ufunc_kwargs = {"n_dims": 1, "n_output": 2, "ravel": False, "check_shape": False}
    kwargs = {"input_core_dims": [["sample"]], "output_core_dims": [["sample"], []]}
    log_weights, pareto_shape = _wrap_xarray_ufunc(
        _psislw, log_weights, ufunc_kwargs=ufunc_kwargs, func_kwargs=func_kwargs, **kwargs
    )
    if isinstance(log_weights, xr.DataArray):
        log_weights = log_weights.rename("log_weights").rename(sample="sample")
    if isinstance(pareto_shape, xr.DataArray):
        pareto_shape = pareto_shape.rename("pareto_shape")
    return log_weights, pareto_shape


def _psislw(log_weights, cutoff_ind, cutoffmin, k_min=1.0 / 3):
    """
    Pareto smoothed importance sampling (PSIS) for a 1D vector.

    Parameters
    ----------
    log_weights : array
        Array of length n_observations
    cutoff_ind : int
    cutoffmin : float
    k_min : float

    Returns
    -------
    lw_out : array
        Smoothed log weights
    kss : float
        Pareto tail index
    """
    x = np.asarray(log_weights)

    # improve numerical accuracy
    x -= np.max(x)
    # sort the array
    x_sort_ind = np.argsort(x)
    # divide log weights into body and right tail
    xcutoff = max(x[x_sort_ind[cutoff_ind]], cutoffmin)

    expxcutoff = np.exp(xcutoff)
    (tailinds,) = np.where(x > xcutoff)  # pylint: disable=unbalanced-tuple-unpacking
    x_tail = x[tailinds]
    tail_len = len(x_tail)
    if tail_len <= 4:
        # not enough tail samples for gpdfit
        k = np.inf
    else:
        # order of tail samples
        x_tail_si = np.argsort(x_tail)
        # fit generalized Pareto distribution to the right tail samples
        x_tail = np.exp(x_tail) - expxcutoff
        k, sigma = _gpdfit(x_tail[x_tail_si])

        if k >= k_min:
            # no smoothing if short tail or GPD fit failed
            # compute ordered statistic for the fit
            sti = np.arange(0.5, tail_len) / tail_len
            smoothed_tail = _gpinv(sti, k, sigma)
            smoothed_tail = np.log(  # pylint: disable=assignment-from-no-return
                smoothed_tail + expxcutoff
            )
            # place the smoothed tail into the output array
            x[tailinds[x_tail_si]] = smoothed_tail
            # truncate smoothed values to the largest raw weight 0
            x[x > 0] = 0
    # renormalize weights
    x -= _logsumexp(x)

    return x, k


def _gpdfit(ary):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD).

    Empirical Bayes estimate for the parameters of the generalized Pareto
    distribution given the data.

    Parameters
    ----------
    ary : array
        sorted 1D data array

    Returns
    -------
    k : float
        estimated shape parameter
    sigma : float
        estimated scale parameter
    """
    prior_bs = 3
    prior_k = 10
    n = len(ary)
    m_est = 30 + int(n ** 0.5)

    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1 / ary[-1]

    k_ary = np.log1p(-b_ary[:, None] * ary).mean(axis=1)  # pylint: disable=no-member
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)

    # remove negligible weights
    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]
    # normalise weights
    weights /= weights.sum()

    # posterior mean for b
    b_post = np.sum(b_ary * weights)
    # estimate for k
    k_post = np.log1p(-b_post * ary).mean()  # pylint: disable=invalid-unary-operand-type,no-member
    # add prior for k_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    sigma = -k_post / b_post

    return k_post, sigma


def _gpinv(probs, kappa, sigma):
    """Inverse Generalized Pareto distribution function."""
    # pylint: disable=unsupported-assignment-operation, invalid-unary-operand-type
    x = np.full_like(probs, np.nan)
    if sigma <= 0:
        return x
    ok = (probs > 0) & (probs < 1)
    if np.all(ok):
        if np.abs(kappa) < np.finfo(float).eps:
            x = -np.log1p(-probs)
        else:
            x = np.expm1(-kappa * np.log1p(-probs)) / kappa
        x *= sigma
    else:
        if np.abs(kappa) < np.finfo(float).eps:
            x[ok] = -np.log1p(-probs[ok])
        else:
            x[ok] = np.expm1(-kappa * np.log1p(-probs[ok])) / kappa
        x *= sigma
        x[probs == 0] = 0
        if kappa >= 0:
            x[probs == 1] = np.inf
        else:
            x[probs == 1] = -sigma / kappa
    return x


def r2_score(y_true, y_pred):
    """R² for Bayesian regression models. Only valid for linear models.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    Pandas Series with the following indices:
    r2: Bayesian R²
    r2_std: standard deviation of the Bayesian R².
    """
    _numba_flag = Numba.numba_flag
    if y_pred.ndim == 1:
        var_y_est = _numba_var(svar, np.var, y_pred)
        var_e = _numba_var(svar, np.var, (y_true - y_pred))
    else:
        var_y_est = _numba_var(svar, np.var, y_pred.mean(0))
        var_e = _numba_var(svar, np.var, (y_true - y_pred), axis=0)
    r_squared = var_y_est / (var_y_est + var_e)

    return pd.Series([np.mean(r_squared), np.std(r_squared)], index=["r2", "r2_std"])


def summary(
    data,
    var_names: Optional[List[str]] = None,
    fmt: str = "wide",
    kind: str = "all",
    round_to=None,
    include_circ=None,
    stat_funcs=None,
    extend=True,
    credible_interval=0.94,
    order="C",
    index_origin=0,
    coords: Optional[CoordSpec] = None,
    dims: Optional[DimSpec] = None,
) -> Union[pd.DataFrame, xr.Dataset]:
    """Create a data frame with summary statistics.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list
        Names of variables to include in summary
    fmt : {'wide', 'long', 'xarray'}
        Return format is either pandas.DataFrame {'wide', 'long'} or xarray.Dataset {'xarray'}.
    kind : {'all', 'stats', 'diagnostics'}
        Whether to include the `stats`: `mean`, `sd`, `hpd_3%`, `hpd_97%`, or the `diagnostics`:
        `mcse_mean`, `mcse_sd`, `ess_bulk`, `ess_tail`, and `r_hat`. Default to include `all` of
        them.
    round_to : int
        Number of decimals used to round results. Defaults to 2. Use "none" to return raw numbers.
    include_circ : bool
        Whether to include circular statistics
    stat_funcs : dict
        A list of functions or a dict of functions with function names as keys used to calculate
        statistics. By default, the mean, standard deviation, simulation standard error, and
        highest posterior density intervals are included.

        The functions will be given one argument, the samples for a variable as an nD array,
        The functions should be in the style of a ufunc and return a single number. For example,
        `np.mean`, or `scipy.stats.var` would both work.
    extend : boolean
        If True, use the statistics returned by `stat_funcs` in addition to, rather than in place
        of, the default statistics. This is only meaningful when `stat_funcs` is not None.
    credible_interval : float, optional
        Credible interval to plot. Defaults to 0.94. This is only meaningful when `stat_funcs` is
        None.
    order : {"C", "F"}
        If fmt is "wide", use either C or F unpacking order. Defaults to C.
    index_origin : int
        If fmt is "wide, select n-based indexing for multivariate parameters. Defaults to 0.
    coords: Dict[str, List[Any]], optional
        Coordinates specification to be used if the ``fmt`` is ``'xarray'``.
    dims: Dict[str, List[str]], optional
        Dimensions specification for the variables to be used if the ``fmt`` is ``'xarray'``.

    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        Return type dicated by `fmt` argument.
        Return value will contain summary statistics for each variable. Default statistics are:
        `mean`, `sd`, `hpd_3%`, `hpd_97%`, `mcse_mean`, `mcse_sd`, `ess_bulk`, `ess_tail`, and
        `r_hat`.
        `r_hat` is only computed for traces with 2 or more chains.

    Examples
    --------
    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("centered_eight")
           ...: az.summary(data, var_names=["mu", "tau"])

    Other statistics can be calculated by passing a list of functions
    or a dictionary with key, function pairs.

    .. ipython::

        In [1]: import numpy as np
           ...: def median_sd(x):
           ...:     median = np.percentile(x, 50)
           ...:     sd = np.sqrt(np.mean((x-median)**2))
           ...:     return sd
           ...:
           ...: func_dict = {
           ...:     "std": np.std,
           ...:     "median_std": median_sd,
           ...:     "5%": lambda x: np.percentile(x, 5),
           ...:     "median": lambda x: np.percentile(x, 50),
           ...:     "95%": lambda x: np.percentile(x, 95),
           ...: }
           ...: az.summary(
           ...:     data,
           ...:     var_names=["mu", "tau"],
           ...:     stat_funcs=func_dict,
           ...:     extend=False
           ...: )

    """
    extra_args = {}  # type: Dict[str, Any]
    if coords is not None:
        extra_args["coords"] = coords
    if dims is not None:
        extra_args["dims"] = dims
    posterior = convert_to_dataset(data, group="posterior", **extra_args)
    var_names = _var_names(var_names, posterior)
    posterior = posterior if var_names is None else posterior[var_names]

    fmt_group = ("wide", "long", "xarray")
    if not isinstance(fmt, str) or (fmt.lower() not in fmt_group):
        raise TypeError("Invalid format: '{}'. Formatting options are: {}".format(fmt, fmt_group))

    unpack_order_group = ("C", "F")
    if not isinstance(order, str) or (order.upper() not in unpack_order_group):
        raise TypeError(
            "Invalid order: '{}'. Unpacking options are: {}".format(order, unpack_order_group)
        )

    alpha = 1 - credible_interval

    extra_metrics = []
    extra_metric_names = []

    if stat_funcs is not None:
        if isinstance(stat_funcs, dict):
            for stat_func_name, stat_func in stat_funcs.items():
                extra_metrics.append(
                    xr.apply_ufunc(
                        _make_ufunc(stat_func), posterior, input_core_dims=(("chain", "draw"),)
                    )
                )
                extra_metric_names.append(stat_func_name)
        else:
            for stat_func in stat_funcs:
                extra_metrics.append(
                    xr.apply_ufunc(
                        _make_ufunc(stat_func), posterior, input_core_dims=(("chain", "draw"),)
                    )
                )
                extra_metric_names.append(stat_func.__name__)

    if extend and kind in ["all", "stats"]:
        mean = posterior.mean(dim=("chain", "draw"))

        sd = posterior.std(dim=("chain", "draw"), ddof=1)

        hpd_lower, hpd_higher = xr.apply_ufunc(
            _make_ufunc(hpd, n_output=2),
            posterior,
            kwargs=dict(credible_interval=credible_interval, multimodal=False),
            input_core_dims=(("chain", "draw"),),
            output_core_dims=tuple([] for _ in range(2)),
        )

    if include_circ:
        circ_mean = xr.apply_ufunc(
            _make_ufunc(st.circmean),
            posterior,
            kwargs=dict(high=np.pi, low=-np.pi),
            input_core_dims=(("chain", "draw"),),
        )
        _numba_flag = Numba.numba_flag
        func = None
        if _numba_flag:
            func = _circular_standard_deviation
        else:
            func = st.circstd
        circ_sd = xr.apply_ufunc(
            _make_ufunc(func),
            posterior,
            kwargs=dict(high=np.pi, low=-np.pi),
            input_core_dims=(("chain", "draw"),),
        )

        circ_mcse = xr.apply_ufunc(
            _make_ufunc(_mc_error),
            posterior,
            kwargs=dict(circular=True),
            input_core_dims=(("chain", "draw"),),
        )

        circ_hpd_lower, circ_hpd_higher = xr.apply_ufunc(
            _make_ufunc(hpd, n_output=2),
            posterior,
            kwargs=dict(credible_interval=credible_interval, circular=True),
            input_core_dims=(("chain", "draw"),),
            output_core_dims=tuple([] for _ in range(2)),
        )

    if kind in ["all", "diagnostics"]:
        mcse_mean, mcse_sd, ess_mean, ess_sd, ess_bulk, ess_tail, r_hat = xr.apply_ufunc(
            _make_ufunc(_multichain_statistics, n_output=7, ravel=False),
            posterior,
            input_core_dims=(("chain", "draw"),),
            output_core_dims=tuple([] for _ in range(7)),
        )

    # Combine metrics
    metrics = []
    metric_names = []
    if extend:
        metrics_names_ = (
            "mean",
            "sd",
            "hpd_{:g}%".format(100 * alpha / 2),
            "hpd_{:g}%".format(100 * (1 - alpha / 2)),
            "mcse_mean",
            "mcse_sd",
            "ess_mean",
            "ess_sd",
            "ess_bulk",
            "ess_tail",
            "r_hat",
        )
        if kind == "all":
            metrics_ = (
                mean,
                sd,
                hpd_lower,
                hpd_higher,
                mcse_mean,
                mcse_sd,
                ess_mean,
                ess_sd,
                ess_bulk,
                ess_tail,
                r_hat,
            )
        elif kind == "stats":
            metrics_ = (mean, sd, hpd_lower, hpd_higher)
            metrics_names_ = metrics_names_[:4]
        elif kind == "diagnostics":
            metrics_ = (mcse_mean, mcse_sd, ess_mean, ess_sd, ess_bulk, ess_tail, r_hat)
            metrics_names_ = metrics_names_[4:]
        metrics.extend(metrics_)
        metric_names.extend(metrics_names_)
    if include_circ:
        metrics.extend((circ_mean, circ_sd, circ_hpd_lower, circ_hpd_higher, circ_mcse))
        metric_names.extend(
            (
                "circular_mean",
                "circular_sd",
                "circular_hpd_{:g}%".format(100 * alpha / 2),
                "circular_hpd_{:g}%".format(100 * (1 - alpha / 2)),
                "circular_mcse",
            )
        )
    metrics.extend(extra_metrics)
    metric_names.extend(extra_metric_names)
    joined = (
        xr.concat(metrics, dim="metric").assign_coords(metric=metric_names).reset_coords(drop=True)
    )

    if fmt.lower() == "wide":
        dfs = []
        for var_name, values in joined.data_vars.items():
            if len(values.shape[1:]):
                metric = list(values.metric.values)
                data_dict = OrderedDict()
                for idx in np.ndindex(values.shape[1:] if order == "C" else values.shape[1:][::-1]):
                    if order == "F":
                        idx = tuple(idx[::-1])
                    ser = pd.Series(values[(Ellipsis, *idx)].values, index=metric)
                    key_index = ",".join(map(str, (i + index_origin for i in idx)))
                    key = "{}[{}]".format(var_name, key_index)
                    data_dict[key] = ser
                df = pd.DataFrame.from_dict(data_dict, orient="index")
                df = df.loc[list(data_dict.keys())]
            else:
                df = values.to_dataframe()
                df.index = list(df.index)
                df = df.T
            dfs.append(df)
        summary_df = pd.concat(dfs, sort=False)
    elif fmt.lower() == "long":
        df = joined.to_dataframe().reset_index().set_index("metric")
        df.index = list(df.index)
        summary_df = df
    else:
        # format is 'xarray'
        summary_df = joined
    if (round_to is not None) and (round_to not in ("None", "none")):
        summary_df = summary_df.round(round_to)
    elif round_to not in ("None", "none") and (fmt.lower() in ("long", "wide")):
        # Don't round xarray object by default (even with "none")
        decimals = {
            col: 3
            if col not in {"ess_mean", "ess_sd", "ess_bulk", "ess_tail", "r_hat"}
            else 2
            if col == "r_hat"
            else 0
            for col in summary_df.columns
        }
        summary_df = summary_df.round(decimals)

    return summary_df


def waic(data, pointwise=False, scale="deviance"):
    """Calculate the widely available information criterion.

    Also calculates the WAIC's standard error and the effective number of
    parameters of the samples in trace from model. Read more theory here - in
    a paper by some of the leading authorities on model selection
    <dx.doi.org/10.1111/1467-9868.00353>

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_inference_data for details
    pointwise : bool
        if True the pointwise predictive accuracy will be returned.
        Default False
    scale : str
        Output scale for loo. Available options are:

        - `deviance` : (default) -2 * (log-score)
        - `log` : 1 * log-score
        - `negative_log` : -1 * (log-score)

        A higher log-score (or a lower deviance) indicates a model with better predictive
        accuracy.

    Returns
    -------
    Series with the following rows:
    waic : widely available information criterion
    waic_se : standard error of waic
    p_waic : effective number parameters
    var_warn : bool
        True if posterior variance of the log predictive
        densities exceeds 0.4
    waic_i : and array of the pointwise predictive accuracy, only if pointwise True
    waic_scale : scale of the waic results

        The returned object has a custom print method that overrides pd.Series method. It is
        specific to expected log pointwise predictive density (elpd) information criteria.

    Examples
    --------
    Calculate the WAIC of a model:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("centered_eight")
           ...: az.waic(data, pointwise=True)

    The custom print method can be seen here, printing only the relevant information and
    with a specific organization. ``IC_loo`` stands for information criteria, which is the
    `deviance` scale, the `log` (and `negative_log`) correspond to ``elpd`` (and ``-elpd``)
    """
    inference_data = convert_to_inference_data(data)
    for group in ("sample_stats",):
        if not hasattr(inference_data, group):
            raise TypeError(
                "Must be able to extract a {group} group from data.".format(group=group)
            )
    if "log_likelihood" not in inference_data.sample_stats:
        raise TypeError("Data must include log_likelihood in sample_stats")
    log_likelihood = inference_data.sample_stats.log_likelihood

    if scale.lower() == "deviance":
        scale_value = -2
    elif scale.lower() == "log":
        scale_value = 1
    elif scale.lower() == "negative_log":
        scale_value = -1
    else:
        raise TypeError('Valid scale values are "deviance", "log", "negative_log"')

    log_likelihood = log_likelihood.stack(sample=("chain", "draw"))
    shape = log_likelihood.shape
    n_samples = shape[-1]
    n_data_points = np.product(shape[:-1])

    ufunc_kwargs = {"n_dims": 1, "ravel": False}
    kwargs = {"input_core_dims": [["sample"]]}
    lppd_i = _wrap_xarray_ufunc(
        _logsumexp,
        log_likelihood,
        func_kwargs={"b_inv": n_samples},
        ufunc_kwargs=ufunc_kwargs,
        **kwargs,
    )

    vars_lpd = log_likelihood.var(dim="sample")
    warn_mg = False
    if np.any(vars_lpd > 0.4):
        warnings.warn(
            (
                "For one or more samples the posterior variance of the log predictive "
                "densities exceeds 0.4. This could be indication of WAIC starting to fail. \n"
                "See http://arxiv.org/abs/1507.04544 for details"
            )
        )
        warn_mg = True

    waic_i = scale_value * (lppd_i - vars_lpd)
    waic_se = (n_data_points * np.var(waic_i.values)) ** 0.5
    waic_sum = np.sum(waic_i.values)
    p_waic = np.sum(vars_lpd.values)

    if pointwise:
        if np.equal(waic_sum, waic_i).all():  # pylint: disable=no-member
            warnings.warn(
                """The point-wise WAIC is the same with the sum WAIC, please double check
            the Observed RV in your model to make sure it returns element-wise logp.
            """
            )
        return ELPDData(
            data=[
                waic_sum,
                waic_se,
                p_waic,
                n_samples,
                n_data_points,
                warn_mg,
                waic_i.rename("waic_i"),
                scale,
            ],
            index=[
                "waic",
                "waic_se",
                "p_waic",
                "n_samples",
                "n_data_points",
                "warning",
                "waic_i",
                "waic_scale",
            ],
        )
    else:
        return ELPDData(
            data=[waic_sum, waic_se, p_waic, n_samples, n_data_points, warn_mg, scale],
            index=[
                "waic",
                "waic_se",
                "p_waic",
                "n_samples",
                "n_data_points",
                "warning",
                "waic_scale",
            ],
        )


def loo_pit(idata=None, *, y=None, y_hat=None, log_weights=None):
    """Compute leave one out (LOO) probability integral transform (PIT) values.

    Parameters
    ----------
    idata : InferenceData
        InferenceData object.
    y : array, DataArray or str
        Observed data. If str, idata must be present and contain the observed data group
    y_hat : array, DataArray or str
        Posterior predictive samples for ``y``. It must have the same shape as y plus an
        extra dimension at the end of size n_samples (chains and draws stacked). If str or
        None, idata must contain the posterior predictive group. If None, y_hat is taken
        equal to y, thus, y must be str too.
    log_weights : array or DataArray
        Smoothed log_weights. It must have the same shape as ``y_hat``

    Returns
    -------
    loo_pit : array or DataArray
        Value of the LOO-PIT at each observed data point.

    Examples
    --------
    Calculate LOO-PIT values using as test quantity the observed values themselves.

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("centered_eight")
           ...: az.loo_pit(idata=data, y="obs")

    Calculate LOO-PIT values using as test quantity the square of the difference between
    each observation and `mu`. Both ``y`` and ``y_hat`` inputs will be array-like,
    but ``idata`` will still be passed in order to calculate the ``log_weights`` from
    there.

    .. ipython::

        In [1]: T = data.observed_data.obs - data.posterior.mu.median(dim=("chain", "draw"))
           ...: T_hat = data.posterior_predictive.obs - data.posterior.mu
           ...: T_hat = T_hat.stack(sample=("chain", "draw"))
           ...: az.loo_pit(idata=data, y=T**2, y_hat=T_hat**2)

    """
    if idata is not None and not isinstance(idata, InferenceData):
        raise ValueError("idata must be of type InferenceData or None")

    if idata is None:
        if not all(isinstance(arg, (np.ndarray, xr.DataArray)) for arg in (y, y_hat, log_weights)):
            raise ValueError(
                "all 3 y, y_hat and log_weights must be array or DataArray when idata is None "
                "but they are of types {}".format([type(arg) for arg in (y, y_hat, log_weights)])
            )

    else:
        if y_hat is None and isinstance(y, str):
            y_hat = y
        elif y_hat is None:
            raise ValueError("y_hat cannot be None if y is not a str")
        if isinstance(y, str):
            y = idata.observed_data[y].values
        elif not isinstance(y, (np.ndarray, xr.DataArray)):
            raise ValueError("y must be of types array, DataArray or str, not {}".format(type(y)))
        if isinstance(y_hat, str):
            y_hat = idata.posterior_predictive[y_hat].stack(sample=("chain", "draw")).values
        elif not isinstance(y_hat, (np.ndarray, xr.DataArray)):
            raise ValueError(
                "y_hat must be of types array, DataArray or str, not {}".format(type(y_hat))
            )
        if log_weights is None:
            log_likelihood = idata.sample_stats.log_likelihood.stack(sample=("chain", "draw"))
            posterior = convert_to_dataset(idata, group="posterior")
            n_chains = len(posterior.chain)
            n_samples = len(log_likelihood.sample)
            ess_p = ess(posterior, method="mean")
            # this mean is over all data variables
            reff = (
                (np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples)
                if n_chains > 1
                else 1
            )
            log_weights = psislw(-log_likelihood, reff=reff)[0].values
        elif not isinstance(log_weights, (np.ndarray, xr.DataArray)):
            raise ValueError(
                "log_weights must be None or of types array or DataArray, not {}".format(
                    type(log_weights)
                )
            )

    if len(y.shape) + 1 != len(y_hat.shape):
        raise ValueError(
            "y_hat must have 1 more dimension than y, but y_hat has {} dims and y has "
            "{} dims".format(len(y.shape), len(y_hat.shape))
        )

    if y.shape != y_hat.shape[:-1]:
        raise ValueError(
            "y has shape: {} which should be equal to y_hat shape (omitting the last "
            "dimension): {}".format(y.shape, y_hat.shape)
        )

    if y_hat.shape != log_weights.shape:
        raise ValueError(
            "y_hat and log_weights must have the same shape but have shapes {} and {}".format(
                y_hat.shape, log_weights.shape
            )
        )

    kwargs = {
        "input_core_dims": [[], ["sample"], ["sample"]],
        "output_core_dims": [[]],
        "join": "left",
    }
    ufunc_kwargs = {"n_dims": 1}

    return _wrap_xarray_ufunc(_loo_pit, y, y_hat, log_weights, ufunc_kwargs=ufunc_kwargs, **kwargs)


def _loo_pit(y, y_hat, log_weights):
    """Compute LOO-PIT values."""
    sel = y_hat <= y
    if np.sum(sel) > 0:
        value = np.exp(_logsumexp(log_weights[sel]))
        return min(1, value)
    else:
        return 0


def apply_test_function(
    idata,
    func,
    group="both",
    var_names=None,
    pointwise=False,
    out_data_shape=None,
    out_pp_shape=None,
    out_name_data="T",
    out_name_pp=None,
    func_args=None,
    func_kwargs=None,
    ufunc_kwargs=None,
    wrap_data_kwargs=None,
    wrap_pp_kwargs=None,
    inplace=True,
    overwrite=None,
):
    """Apply a Bayesian test function to an InferenceData object.

    Parameters
    ----------
    idata : InferenceData
        InferenceData object on which to apply the test function. This function will add
        new variables to the InferenceData object to store the result without modifying the
        existing ones.
    func : callable
        Callable that calculates the test function. It must have the following call signature
        ``func(y, theta, *args, **kwargs)`` (where ``y`` is the observed data or posterior
        predictive and ``theta`` the model parameters) even if not all the arguments are
        used.
    group : str, optional
        Group on which to apply the test function. Can be observed_data, posterior_predictive
        or both.
    var_names : dict group -> var_names, optional
        Mapping from group name to the variables to be passed to func. It can be a dict of
        strings or lists of strings. There is also the option of using ``both`` as key,
        in which case, the same variables are used in observed data and posterior predictive
        groups
    pointwise : bool, optional
        If True, apply the test function to each observation and sample, otherwise, apply
        test function to each sample.
    out_data_shape, out_pp_shape : tuple, optional
        Output shape of the test function applied to the observed/posterior predictive data.
        If None, the default depends on the value of pointwise.
    out_name_data, out_name_pp : str, optional
        Name of the variables to add to the observed_data and posterior_predictive datasets
        respectively. ``out_name_pp`` can be ``None``, in which case will be taken equal to
        ``out_name_data``.
    func_args : sequence, optional
        Passed as is to ``func``
    func_kwargs : mapping, optional
        Passed as is to ``func``
    wrap_data_kwargs, wrap_pp_kwargs : mapping, optional
        kwargs passed to ``az.stats.wrap_xarray_ufunc``. By default, some suitable input_core_dims
        are used.
    inplace : bool, optional
        If True, add the variables inplace, othewise, return a copy of idata with the variables
        added.
    overwrite : bool, optional
        Overwrite data in case ``out_name_data`` or ``out_name_pp`` are already variables in
        dataset. If ``None`` it will be the opposite of inplace.

    Returns
    -------
    idata : InferenceData
        Output InferenceData object. If ``inplace=True``, it is the same input object modified
        inplace.

    Notes
    -----
    This function is provided for convenience to wrap scalar or functions working on low
    dims to inference data object. It is not optimized to be faster nor as fast as vectorized
    computations.

    Examples
    --------
    Use ``apply_test_function`` to wrap ``np.min`` for illustration purposes. And plot the
    results.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> idata = az.load_arviz_data("centered_eight")
        >>> az.apply_test_function(idata, lambda y, theta: np.min(y))
        >>> T = np.asscalar(idata.observed_data.T)
        >>> az.plot_posterior(idata, var_names=["T"], group="posterior_predictive", ref_val=T)

    """
    out = idata if inplace else deepcopy(idata)

    valid_groups = ("observed_data", "posterior_predictive", "both")
    if group not in valid_groups:
        raise ValueError(
            "Invalid group argument. Must be one of {} not {}.".format(valid_groups, group)
        )
    if overwrite is None:
        overwrite = not inplace

    if out_name_pp is None:
        out_name_pp = out_name_data

    if func_args is None:
        func_args = tuple()

    if func_kwargs is None:
        func_kwargs = {}

    if ufunc_kwargs is None:
        ufunc_kwargs = {}
    ufunc_kwargs.setdefault("check_shape", False)
    ufunc_kwargs.setdefault("ravel", False)

    if wrap_data_kwargs is None:
        wrap_data_kwargs = {}
    if wrap_pp_kwargs is None:
        wrap_pp_kwargs = {}
    if var_names is None:
        var_names = {}

    both_var_names = var_names.pop("both", None)
    var_names.setdefault("posterior", list(out.posterior.data_vars))

    in_posterior = out.posterior[var_names["posterior"]]
    if isinstance(in_posterior, xr.Dataset):
        in_posterior = in_posterior.to_array().squeeze()

    groups = ("posterior_predictive", "observed_data") if group == "both" else [group]
    for grp in groups:
        out_group_shape = out_data_shape if grp == "observed_data" else out_pp_shape
        out_name_group = out_name_data if grp == "observed_data" else out_name_pp
        wrap_group_kwargs = wrap_data_kwargs if grp == "observed_data" else wrap_pp_kwargs
        if not hasattr(out, grp):
            raise ValueError("InferenceData object must have {} group".format(grp))
        if not overwrite and out_name_group in getattr(out, grp).data_vars:
            raise ValueError(
                "Should overwrite: {} variable present in group {}, but overwrite is False".format(
                    out_name_group, grp
                )
            )
        var_names.setdefault(
            grp, list(getattr(out, grp).data_vars) if both_var_names is None else both_var_names
        )
        in_group = getattr(out, grp)[var_names[grp]]
        if isinstance(in_group, xr.Dataset):
            in_group = in_group.to_array(dim="{}_var".format(grp)).squeeze()

        if pointwise:
            out_group_shape = in_group.shape if out_group_shape is None else out_group_shape
        elif grp == "observed_data":
            out_group_shape = () if out_group_shape is None else out_group_shape
        elif grp == "posterior_predictive":
            out_group_shape = in_group.shape[:2] if out_group_shape is None else out_group_shape
        loop_dims = in_group.dims[: len(out_group_shape)]

        wrap_group_kwargs.setdefault(
            "input_core_dims",
            [
                [dim for dim in dataset.dims if dim not in loop_dims]
                for dataset in [in_group, in_posterior]
            ],
        )
        func_kwargs["out"] = np.empty(out_group_shape)

        out_group = getattr(out, grp)
        try:
            out_group[out_name_group] = _wrap_xarray_ufunc(
                func,
                in_group.values,
                in_posterior.values,
                func_args=func_args,
                func_kwargs=func_kwargs,
                ufunc_kwargs=ufunc_kwargs,
                **wrap_group_kwargs,
            )
        except IndexError:
            excluded_dims = set(
                wrap_group_kwargs["input_core_dims"][0] + wrap_group_kwargs["input_core_dims"][1]
            )
            out_group[out_name_group] = _wrap_xarray_ufunc(
                func,
                *xr.broadcast(in_group, in_posterior, exclude=excluded_dims),
                func_args=func_args,
                func_kwargs=func_kwargs,
                ufunc_kwargs=ufunc_kwargs,
                **wrap_group_kwargs,
            )
        setattr(out, grp, out_group)

    return out
