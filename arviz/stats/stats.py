"""Statistical functions in ArviZ."""
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import scipy.stats as st
from scipy.optimize import minimize
import xarray as xr

from arviz.data import convert_to_inference_data, convert_to_dataset
from .diagnostics import effective_n, gelman_rubin

__all__ = ["bfmi", "compare", "hpd", "loo", "psislw", "r2_score", "summary", "waic"]


def bfmi(energy):
    r"""Calculate the estimated Bayesian fraction of missing information (BFMI).

    BFMI quantifies how well momentum resampling matches the marginal energy distribution. For more
    information on BFMI, see https://arxiv.org/pdf/1604.00695v1.pdf. The current advice is that
    values smaller than 0.3 indicate poor sampling. However, this threshold is provisional and may
    change.  See http://mc-stan.org/users/documentation/case-studies/pystan_workflow.html for more
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
    return np.square(np.diff(energy_mat, axis=1)).mean(axis=1) / np.var(
        energy_mat, axis=1
    )


def compare(dataset_dict, ic="waic", method="stacking", b_samples=1000, alpha=1,
            seed=None):
    r"""Compare models based on WAIC or LOO cross validation.

    WAIC is Widely applicable information criterion, and LOO is leave-one-out
    (LOO) cross-validation. Read more theory here - in a paper by some of the
    leading authorities on model selection - dx.doi.org/10.1111/1467-9868.00353

    Parameters
    ----------
    dataset_dict : dict[str] -> InferenceData
        A dictionary of model names and InferenceData objects
    ic : string
        Information Criterion (WAIC or LOO) used to compare models. Default WAIC.
    method : str
        Method used to estimate the weights for each model. Available options are:

        - 'stacking' : (default) stacking of predictive distributions.
        - 'BB-pseudo-BMA' : pseudo-Bayesian Model averaging using Akaike-type
           weighting. The weights are stabilized using the Bayesian bootstrap
        - 'pseudo-BMA': pseudo-Bayesian Model averaging using Akaike-type
           weighting, without Bootstrap stabilization (not recommended)

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

    Returns
    -------
    A DataFrame, ordered from lowest to highest IC. The index reflects the order in which the
    models are passed to this function. The columns are:
    IC : Information Criteria (WAIC or LOO).
        Smaller IC indicates higher out-of-sample predictive fit ("better" model). Default WAIC.
    pIC : Estimated effective number of parameters.
    dIC : Relative difference between each IC (WAIC or LOO)
    and the lowest IC (WAIC or LOO).
        It's always 0 for the top-ranked model.
    weight: Relative weight for each model.
        This can be loosely interpreted as the probability of each model (among the compared model)
        given the data. By default the uncertainty in the weights estimation is considered using
        Bayesian bootstrap.
    SE : Standard error of the IC estimate.
        If method = BB-pseudo-BMA these values are estimated using Bayesian bootstrap.
    dSE : Standard error of the difference in IC between each model and
    the top-ranked model.
        It's always 0 for the top-ranked model.
    warning : A value of 1 indicates that the computation of the IC may not be reliable. This could
        be indication of WAIC/LOO starting to fail see http://arxiv.org/abs/1507.04544 for details.
    """
    names = list(dataset_dict.keys())

    if ic == "waic":
        ic_func = waic
        df_comp = pd.DataFrame(
            index=names,
            columns=["waic", "pwaic", "dwaic", "weight", "se", "dse", "warning"],
        )

    elif ic == "loo":
        ic_func = loo
        df_comp = pd.DataFrame(
            index=names,
            columns=["loo", "ploo", "dloo", "weight", "se", "dse", "warning"],
        )

    else:
        raise NotImplementedError(
            "The information criterion {} is not supported.".format(ic)
        )

    if method not in ["stacking", "BB-pseudo-BMA", "pseudo-BMA"]:
        raise ValueError(
            "The method {}, to compute weights, is not supported.".format(method)
        )

    ic_se = "{}_se".format(ic)
    p_ic = "p_{}".format(ic)
    ic_i = "{}_i".format(ic)

    ics = pd.DataFrame()
    names = []
    for name, dataset in dataset_dict.items():
        names.append(name)
        ics = ics.append(ic_func(dataset, pointwise=True))
    ics.index = names
    ics.sort_values(by=ic, inplace=True)

    if method == "stacking":
        rows, cols, ic_i_val = _ic_matrix(ics, ic_i)
        exp_ic_i = np.exp(-0.5 * ic_i_val)
        last_col = cols - 1

        def w_fuller(weights):
            return np.concatenate((weights, [max(1. - np.sum(weights), 0.)]))

        def log_score(weights):
            w_full = w_fuller(weights)
            score = 0.
            for i in range(rows):
                score += np.log(np.dot(exp_ic_i[i], w_full))
            return -score

        def gradient(weights):
            w_full = w_fuller(weights)
            grad = np.zeros(last_col)
            for k in range(last_col):
                for i in range(rows):
                    grad[k] += (exp_ic_i[i, k] - exp_ic_i[i, last_col]) / np.dot(
                        exp_ic_i[i], w_full
                    )
            return -grad

        theta = np.full(last_col, 1. / cols)
        bounds = [(0., 1.) for i in range(last_col)]
        constraints = [
            {"type": "ineq", "fun": lambda x: 1. - np.sum(x)},
            {"type": "ineq", "fun": np.sum},
        ]

        weights = minimize(
            fun=log_score,
            x0=theta,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
        )

        weights = w_fuller(weights["x"])
        ses = ics[ic_se]

    elif method == "BB-pseudo-BMA":
        rows, cols, ic_i_val = _ic_matrix(ics, ic_i)
        ic_i_val = ic_i_val * rows

        b_weighting = st.dirichlet.rvs(
            alpha=[alpha] * rows, size=b_samples, random_state=seed
        )
        weights = np.zeros((b_samples, cols))
        z_bs = np.zeros_like(weights)
        for i in range(b_samples):
            z_b = np.dot(b_weighting[i], ic_i_val)
            u_weights = np.exp(-0.5 * (z_b - np.min(z_b)))
            z_bs[i] = z_b
            weights[i] = u_weights / np.sum(u_weights)

        weights = weights.mean(axis=0)
        ses = pd.Series(z_bs.std(axis=0), index=names)

    elif method == "pseudo-BMA":
        min_ic = ics.iloc[0][ic]
        z_rv = np.exp(-0.5 * (ics[ic] - min_ic))
        weights = z_rv / np.sum(z_rv)
        ses = ics[ic_se]

    if np.any(weights):
        min_ic_i_val = ics[ic_i].iloc[0]
        for idx, val in enumerate(ics.index):
            res = ics.loc[val]
            diff = res[ic_i] - min_ic_i_val
            d_ic = np.sum(diff)
            d_std_err = np.sqrt(len(diff) * np.var(diff))
            std_err = ses.loc[val]
            weight = weights[idx]
            df_comp.at[val] = (
                res[ic],
                res[p_ic],
                d_ic,
                weight,
                std_err,
                d_std_err,
                res["warning"],
            )

    return df_comp.sort_values(by=ic)


def _ic_matrix(ics, ic_i):
    """Store the previously computed pointwise predictive accuracy values (ics) in a 2D matrix."""
    cols, _ = ics.shape
    rows = len(ics[ic_i].iloc[0])
    ic_i_val = np.zeros((rows, cols))

    for idx, val in enumerate(ics.index):
        ic = ics.loc[val][ic_i]
        if len(ic) != rows:
            raise ValueError(
                "The number of observations should be the same across all models"
            )
        else:
            ic_i_val[:, idx] = ic

    return rows, cols, ic_i_val


def hpd(x, credible_interval=0.94, transform=lambda x: x, circular=False):
    """
    Calculate highest posterior density (HPD) of array for given credible_interval.

    The HPD is the minimum width Bayesian credible interval (BCI). This implementation works only
    for unimodal distributions.

    Parameters
    ----------
    x : Numpy array
        An array containing posterior samples
    credible_interval : float, optional
        Credible interval to plot. Defaults to 0.94.
    transform : callable
        Function to transform data (defaults to identity)
    circular : bool, optional
        Whether to compute the error taking into account `x` is a circular variable
        (in the range [-np.pi, np.pi]) or not. Defaults to False (i.e non-circular variables).

    Returns
    -------
    tuple
        lower and upper value of the interval.
    """
    if x.ndim > 1:
        return np.array(
            [
                hpd(
                    row,
                    credible_interval=credible_interval,
                    transform=transform,
                    circular=circular,
                )
                for row in x
            ]
        )
    # Make a copy of trace
    x = transform(x.copy())
    len_x = len(x)

    if circular:
        mean = st.circmean(x, high=np.pi, low=-np.pi)
        x = x - mean
        x = np.arctan2(np.sin(x), np.cos(x))

    x = np.sort(x)
    interval_idx_inc = int(np.floor(credible_interval * len_x))
    n_intervals = len_x - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError(
            "Too few elements for interval calculation."
            " Check that credible_interval meets condition 0 =< credible_interval < 1"
        )

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]

    if circular:
        hdi_min = hdi_min + mean
        hdi_max = hdi_max + mean
        hdi_min = np.arctan2(np.sin(hdi_min), np.cos(hdi_min))
        hdi_max = np.arctan2(np.sin(hdi_max), np.cos(hdi_max))

    return np.array([hdi_min, hdi_max])


def loo(data, pointwise=False, reff=None):
    """Pareto-smoothed importance sampling leave-one-out cross-validation.

    Calculates leave-one-out (LOO) cross-validation for out of sample predictive model fit,
    following Vehtari et al. (2015). Cross-validation is computed using Pareto-smoothed
    importance sampling (PSIS).

    Parameters
    ----------
    data : result of MCMC run
    pointwise: bool, optional
        if True the pointwise predictive accuracy will be returned. Defaults to False
    reff : float, optional
        Relative MCMC efficiency, `effective_n / n` i.e. number of effective samples divided by
        the number of actual samples. Computed from trace by default.

    Returns
    -------
    DataFrame with the following columns:
    loo: approximated Leave-one-out cross-validation
    loo_se: standard error of loo
    p_loo: effective number of parameters
    shape_warn: 1 if the estimated shape parameter of
        Pareto distribution is greater than 0.7 for one or more samples
    loo_i: array of pointwise predictive accuracy, only if pointwise True
    """
    inference_data = convert_to_inference_data(data)
    for group in ("posterior", "sample_stats"):
        if not hasattr(inference_data, group):
            raise TypeError(
                "Must be able to extract a {group} group from data!".format(group=group)
            )
    if "log_likelihood" not in inference_data.sample_stats:
        raise TypeError("Data must include log_likelihood in sample_stats")
    posterior = inference_data.posterior
    log_likelihood = inference_data.sample_stats.log_likelihood
    n_samples = log_likelihood.chain.size * log_likelihood.draw.size
    new_shape = (n_samples,) + log_likelihood.shape[2:]
    log_likelihood = log_likelihood.values.reshape(*new_shape)

    if reff is None:
        n_chains = len(posterior.chain)
        if n_chains == 1:
            reff = 1.
        else:
            eff_n = effective_n(posterior)
            # this mean is over all data variables
            reff = (
                np.hstack([eff_n[v].values.flatten() for v in eff_n.data_vars]).mean()
                / n_samples
            )

    log_weights, pareto_shape = psislw(-log_likelihood, reff)
    log_weights += log_likelihood

    warn_mg = 0
    if np.any(pareto_shape > 0.7):
        warnings.warn(
            """Estimated shape parameter of Pareto distribution is greater than 0.7 for
        one or more samples. You should consider using a more robust model, this is because
        importance sampling is less likely to work well if the marginal posterior and LOO posterior
        are very different. This is more likely to happen with a non-robust model and highly
        influential observations."""
        )
        warn_mg = 1

    loo_lppd_i = -2 * logsumexp(log_weights, axis=0)
    loo_lppd = loo_lppd_i.sum()
    loo_lppd_se = (len(loo_lppd_i) * np.var(loo_lppd_i)) ** 0.5

    lppd = np.sum(logsumexp(log_likelihood, axis=0, b=1. / log_likelihood.shape[0]))
    p_loo = lppd + (0.5 * loo_lppd)

    if pointwise:
        if np.equal(loo_lppd, loo_lppd_i).all():
            warnings.warn(
                """The point-wise LOO is the same with the sum LOO, please double check
            the Observed RV in your model to make sure it returns element-wise logp.
            """
            )
        return pd.DataFrame(
            [[loo_lppd, loo_lppd_se, p_loo, warn_mg, loo_lppd_i]],
            columns=["loo", "loo_se", "p_loo", "warning", "loo_i"],
        )
    else:
        return pd.DataFrame(
            [[loo_lppd, loo_lppd_se, p_loo, warn_mg]],
            columns=["loo", "loo_se", "p_loo", "warning"],
        )


def psislw(log_weights, reff=1.):
    """
    Pareto smoothed importance sampling (PSIS).

    Parameters
    ----------
    log_weights : array
        Array of size (n_samples, n_observations)
    reff : float
        relative MCMC efficiency, `effective_n / n`

    Returns
    -------
    lw_out : array
        Smoothed log weights
    kss : array
        Pareto tail indices
    """
    rows, cols = log_weights.shape

    log_weights_out = np.copy(log_weights, order="F")
    kss = np.empty(cols)

    # precalculate constants
    cutoff_ind = -int(np.ceil(min(rows / 5., 3 * (rows / reff) ** 0.5))) - 1
    cutoffmin = np.log(np.finfo(float).tiny)  # pylint: disable=no-member
    k_min = 1. / 3

    # loop over sets of log weights
    for i, x in enumerate(log_weights_out.T):
        # improve numerical accuracy
        x -= np.max(x)
        # sort the array
        x_sort_ind = np.argsort(x)
        # divide log weights into body and right tail
        xcutoff = max(x[x_sort_ind[cutoff_ind]], cutoffmin)

        expxcutoff = np.exp(xcutoff)
        tailinds, = np.where(x > xcutoff)
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
                smoothed_tail = np.log(smoothed_tail + expxcutoff)
                # place the smoothed tail into the output array
                x[tailinds[x_tail_si]] = smoothed_tail
                # truncate smoothed values to the largest raw weight 0
                x[x > 0] = 0
        # renormalize weights
        x -= logsumexp(x)
        # store tail index k
        kss[i] = k

    return log_weights_out, kss


def _gpdfit(x):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD).

    Empirical Bayes estimate for the parameters of the generalized Pareto
    distribution given the data.

    Parameters
    ----------
    x : array
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
    len_x = len(x)
    m_est = 30 + int(len_x ** 0.5)

    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * x[int(len_x / 4 + 0.5) - 1]
    b_ary += 1 / x[-1]

    k_ary = np.log1p(-b_ary[:, None] * x).mean(axis=1)
    len_scale = len_x * (np.log(-(b_ary / k_ary)) - k_ary - 1)
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
    k_post = np.log1p(-b_post * x).mean()  # pylint: disable=invalid-unary-operand-type
    # add prior for k_post
    k_post = (len_x * k_post + prior_k * 0.5) / (len_x + prior_k)
    sigma = -k_post / b_post

    return k_post, sigma


def _gpinv(probs, kappa, sigma):
    """Inverse Generalized Pareto distribution function."""
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
    y_true: : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    Pandas Series with the following indices:
    r2: Bayesian R²
    r2_std: standard deviation of the Bayesian R².
    """
    if y_pred.ndim == 1:
        var_y_est = np.var(y_pred)
        var_e = np.var(y_true - y_pred)
    else:
        var_y_est = np.var(y_pred.mean(0))
        var_e = np.var(y_true - y_pred, 0)

    r_squared = var_y_est / (var_y_est + var_e)

    return pd.Series(
        [np.mean(r_squared), np.std(r_squared)], index=["r2", "r2_std"]
    )


def summary(data, var_names=None, include_circ=None, stat_funcs=None,
            extend=True, credible_interval=0.94, batches=None):
    """Create a data frame with summary statistics.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list
        Names of variables to include in summary
    include_circ : bool
        Whether to include circular statistics
    stat_funcs : None or list
        A list of functions used to calculate statistics. By default, the mean, standard deviation,
        simulation standard error, and highest posterior density intervals are included.

        The functions will be given one argument, the samples for a variable as an nD array,
        The functions should be in the style of a ufunc and return a single number. For example,
        `np.sin`, or `scipy.stats.var` would both work.
    extend : boolean
        If True, use the statistics returned by `stat_funcs` in addition to, rather than in place
        of, the default statistics. This is only meaningful when `stat_funcs` is not None.
    credible_interval : float, optional
        Credible interval to plot. Defaults to 0.94. This is only meaningful when `stat_funcs` is
        None.
    batches : None or int
        Batch size for calculating standard deviation for non-independent samples. Defaults to the
        smaller of 100 or the number of samples. This is only meaningful when `stat_funcs` is None.

    Returns
    -------
    pd.DataFrame
        With summary statistics for each variable. Defaults statistics are: `mean`, `sd`,
        `hpd_3`, `hpd_97`, `mc_error`, `n_eff` and `Rhat`. `n_eff` and `Rhat` are only computed
        for traces with 2 or more chains.

    Examples
    --------
    .. code:: ipython

        >>> az.summary(trace, ['mu'])
               mean    sd  hpd_3  hpd_97  mc_error  n_eff  Rhat
        mu__0  0.10  0.06  -0.02    0.23      0.00  487.0  1.00
        mu__1 -0.04  0.06  -0.17    0.08      0.00  379.0  1.00

    Other statistics can be calculated by passing a list of functions.

    .. code:: ipython

        >>> import pandas as pd
        >>> def trace_sd(x):
        ...     return pd.Series(np.std(x, 0), name='sd')
        ...
        >>> def trace_quantiles(x):
        ...     return pd.DataFrame(pd.quantiles(x, [5, 50, 95]))
        ...
        >>> az.summary(trace, ['mu'], stat_funcs=[trace_sd, trace_quantiles])
                 sd     5    50    95
        mu__0  0.06  0.00  0.10  0.21
        mu__1  0.07 -0.16 -0.04  0.06
    """
    posterior = convert_to_dataset(data, group="posterior")

    if var_names is None:
        var_names = list(posterior.data_vars)

    if batches is None:
        batches = min([100, posterior.draw.size])

    alpha = 1 - credible_interval

    def make_stat_ufunc(func, **kwargs):
        def circ_ufunc(ary):
            return func(ary.reshape(*ary.shape[:-2], -1), **kwargs, axis=-1)

        return circ_ufunc

    def make_mc_error_ufunc(**kwargs):
        def mc_error_ufunc(ary):
            return _mc_error(ary.reshape(*ary.shape[:-2], -1).T, **kwargs)

        return mc_error_ufunc

    def make_hpd_ufunc(idx, **kwargs):
        def hpd_ufunc(ary):
            return hpd(ary.reshape(*ary.shape[:-2], -1), **kwargs)[..., idx]

        return hpd_ufunc

    metrics = []
    metric_names = []

    if stat_funcs is not None:
        for stat_func in stat_funcs:
            metrics.append(
                xr.apply_ufunc(
                    make_stat_ufunc(stat_func),
                    posterior[var_names],
                    input_core_dims=(("chain", "draw"),),
                )
            )
            metric_names.append(stat_func.__name__)

    if extend:
        metrics.append(posterior[var_names].mean(dim=("chain", "draw")))
        metric_names.append("mean")

        metrics.append(posterior[var_names].std(dim=("chain", "draw")))
        metric_names.append("standard deviation")

        metrics.append(
            xr.apply_ufunc(
                make_mc_error_ufunc(),
                posterior[var_names],
                input_core_dims=(("chain", "draw"),),
            )
        )
        metric_names.append("mc error")

        metrics.append(
            xr.apply_ufunc(
                make_hpd_ufunc(0, credible_interval=credible_interval),
                posterior[var_names],
                input_core_dims=(("chain", "draw"),),
            )
        )
        metric_names.append("hpd {:.2%}".format(alpha / 2))

        metrics.append(
            xr.apply_ufunc(
                make_hpd_ufunc(1, credible_interval=credible_interval),
                posterior[var_names],
                input_core_dims=(("chain", "draw"),),
            )
        )
        metric_names.append("hpd {:.2%}".format(1 - alpha / 2))

    if include_circ:
        metrics.append(
            xr.apply_ufunc(
                make_stat_ufunc(st.circmean, high=np.pi, low=-np.pi),
                posterior[var_names],
                input_core_dims=(("chain", "draw"),),
            )
        )
        metric_names.append("circular mean")

        metrics.append(
            xr.apply_ufunc(
                make_stat_ufunc(st.circmean, high=np.pi, low=-np.pi),
                posterior[var_names],
                input_core_dims=(("chain", "draw"),),
            )
        )
        metric_names.append("circular standard deviation")

        metrics.append(
            xr.apply_ufunc(
                make_mc_error_ufunc(circular=True),
                posterior[var_names],
                input_core_dims=(("chain", "draw"),),
            )
        )
        metric_names.append("circular mc error")

        metrics.append(
            xr.apply_ufunc(
                make_hpd_ufunc(0, credible_interval=credible_interval, circular=True),
                posterior[var_names],
                input_core_dims=(("chain", "draw"),),
            )
        )
        metric_names.append("circular hpd {:.2%}".format(alpha / 2))

        metrics.append(
            xr.apply_ufunc(
                make_hpd_ufunc(1, credible_interval=credible_interval, circular=True),
                posterior[var_names],
                input_core_dims=(("chain", "draw"),),
            )
        )
        metric_names.append("circular hpd {:.2%}".format(1 - alpha / 2))

    metrics.append(effective_n(posterior, var_names=var_names))
    metric_names.append("effective samples")

    metrics.append(gelman_rubin(posterior, var_names=var_names))
    metric_names.append("gelman-rubin statistic")

    joined = xr.concat(metrics, dim="metric").assign_coords(metric=metric_names)
    return joined


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

        try:
            batched_traces = np.resize(x, (batches, int(len(x) / batches)))
        except ValueError:
            # If batches do not divide evenly, trim excess samples
            resid = len(x) % batches
            new_shape = (batches, (len(x) - resid) / batches)
            batched_traces = np.resize(x[:-resid], new_shape)

        if circular:
            means = st.circmean(batched_traces, high=np.pi, low=-np.pi, axis=1)
            std = st.circstd(means, high=np.pi, low=-np.pi)
        else:
            means = np.mean(batched_traces, 1)
            std = np.std(means)

        return std / np.sqrt(batches)


def waic(data, pointwise=False):
    """Calculate the widely available information criterion.

    Also calculates the WAIC's standard error and the effective number of
    parameters of the samples in trace from model. Read more theory here - in
    a paper by some of the leading authorities on model selection
    dx.doi.org/10.1111/1467-9868.00353

    Parameters
    ----------
    pointwise: bool
        if True the pointwise predictive accuracy will be returned.
        Default False

    Returns
    -------
    DataFrame with the following columns:
    waic: widely available information criterion
    waic_se: standard error of waic
    p_waic: effective number parameters
    var_warn: 1 if posterior variance of the log predictive
         densities exceeds 0.4
    waic_i: and array of the pointwise predictive accuracy, only if pointwise True
    """
    inference_data = convert_to_inference_data(data)
    for group in ("sample_stats",):
        if not hasattr(inference_data, group):
            raise TypeError(
                "Must be able to extract a {group} group from data!".format(group=group)
            )
    if "log_likelihood" not in inference_data.sample_stats:
        raise TypeError("Data must include log_likelihood in sample_stats")
    log_likelihood = inference_data.sample_stats.log_likelihood
    n_samples = log_likelihood.chain.size * log_likelihood.draw.size
    new_shape = (n_samples,) + log_likelihood.shape[2:]
    log_likelihood = log_likelihood.values.reshape(*new_shape)

    lppd_i = logsumexp(log_likelihood, axis=0, b=1.0 / log_likelihood.shape[0])

    vars_lpd = np.var(log_likelihood, axis=0)
    warn_mg = 0
    if np.any(vars_lpd > 0.4):
        warnings.warn(
            """For one or more samples the posterior variance of the log predictive
        densities exceeds 0.4. This could be indication of WAIC starting to fail see
        http://arxiv.org/abs/1507.04544 for details
        """
        )
        warn_mg = 1

    waic_i = -2 * (lppd_i - vars_lpd)
    waic_se = (len(waic_i) * np.var(waic_i)) ** 0.5
    waic_sum = np.sum(waic_i)
    p_waic = np.sum(vars_lpd)

    if pointwise:
        if np.equal(waic_sum, waic_i).all():
            warnings.warn(
                """The point-wise WAIC is the same with the sum WAIC, please double check
            the Observed RV in your model to make sure it returns element-wise logp.
            """
            )
        return pd.DataFrame(
            [[waic_sum, waic_se, p_waic, warn_mg, waic_i]],
            columns=["waic", "waic_se", "p_waic", "warning", "waic_i"],
        )
    else:
        return pd.DataFrame(
            [[waic_sum, waic_se, p_waic, warn_mg]],
            columns=["waic", "waic_se", "p_waic", "warning"],
        )
