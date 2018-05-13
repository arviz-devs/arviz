import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import dirichlet, circmean, circstd
from scipy.optimize import minimize

from ..utils import get_stats, get_varnames, trace_to_dataframe, log_post_trace
from .diagnostics import effective_n, gelman_rubin

__all__ = ['bfmi', 'compare', 'hpd', 'loo', 'psislw', 'r2_score', 'summary', 'waic']


def bfmi(trace):
    R"""Calculate the estimated Bayesian fraction of missing information (BFMI).

    BFMI quantifies how well momentum resampling matches the marginal energy distribution. For more
    information on BFMI, see https://arxiv.org/pdf/1604.00695v1.pdf. The current advice is that
    values smaller than 0.3 indicate poor sampling. However, this threshold is provisional and may
    change.  See http://mc-stan.org/users/documentation/case-studies/pystan_workflow.html for more
    information.

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Result of an HMC/NUTS run, must contain energy information

    Returns
    -------
    z : array
        The Bayesian fraction of missing information of the model and trace. One element per
        chain in the trace.
    """
    energy = np.atleast_2d(get_stats(trace, 'energy', combined=False))

    return np.square(np.diff(energy, 1)).mean(1) / np.var(energy, 1)


def compare(model_dict, ic='waic', method='stacking', b_samples=1000,
            alpha=1, seed=None, round_to=2):
    R"""
    Compare models based on the widely applicable information criterion (WAIC) or leave-one-out
    (LOO) cross-validation.

    Read more theory here - in a paper by some of the leading authorities on model selection
    - dx.doi.org/10.1111/1467-9868.00353

    Parameters
    ----------
    model_dict : dictionary of PyMC3 traces indexed by corresponding model
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
    round_to : int
        Number of decimals used to round results (default 2).

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

    names = [model.name for model in model_dict if model.name]
    if not names:
        names = np.arange(len(model_dict))

    if ic == 'waic':
        ic_func = waic
        df_comp = pd.DataFrame(index=names,
                               columns=['waic', 'pwaic', 'dwaic', 'weight','se', 'dse', 'warning'])

    elif ic == 'loo':
        ic_func = loo
        df_comp = pd.DataFrame(index=names,
                               columns=['loo', 'ploo', 'dloo', 'weight', 'se', 'dse', 'warning'])

    else:
        raise NotImplementedError('The information criterion {} is not supported.'.format(ic))

    if len(set([len(m.observed_RVs) for m in model_dict])) != 1:
        raise ValueError('The number of observed RVs should be the same across all models')

    if method not in ['stacking', 'BB-pseudo-BMA', 'pseudo-BMA']:
        raise ValueError('The method {}, to compute weights, is not supported.'.format(method))

    ic_se = '{}_se'.format(ic)
    p_ic = 'p_{}'.format(ic)
    ic_i = '{}_i'.format(ic)

    ics = pd.DataFrame()
    for m, t in model_dict.items():
        ics = ics.append(ic_func(t, m, pointwise=True))
    ics.index = names
    ics.sort_values(by=ic, inplace=True)

    if method == 'stacking':
        N, K, ic_i_val = _ic_matrix(ics, ic_i)
        exp_ic_i = np.exp(-0.5 * ic_i_val)
        Km = K - 1

        def w_fuller(w):
            return np.concatenate((w, [max(1. - np.sum(w), 0.)]))

        def log_score(w):
            w_full = w_fuller(w)
            score = 0.
            for i in range(N):
                score += np.log(np.dot(exp_ic_i[i], w_full))
            return -score

        def gradient(w):
            w_full = w_fuller(w)
            grad = np.zeros(Km)
            for k in range(Km):
                for i in range(N):
                    grad[k] += (exp_ic_i[i, k] - exp_ic_i[i, Km]) / \
                        np.dot(exp_ic_i[i], w_full)
            return -grad

        theta = np.full(Km, 1. / K)
        bounds = [(0., 1.) for i in range(Km)]
        constraints = [{'type': 'ineq', 'fun': lambda x: 1. - np.sum(x)},
                       {'type': 'ineq', 'fun': np.sum}]

        w = minimize(fun=log_score,
                     x0=theta,
                     jac=gradient,
                     bounds=bounds,
                     constraints=constraints)

        weights = w_fuller(w['x'])
        ses = ics[ic_se]

    elif method == 'BB-pseudo-BMA':
        N, K, ic_i_val = _ic_matrix(ics, ic_i)
        ic_i_val = ic_i_val * N

        b_weighting = dirichlet.rvs(alpha=[alpha] * N, size=b_samples,
                                    random_state=seed)
        weights = np.zeros((b_samples, K))
        z_bs = np.zeros_like(weights)
        for i in range(b_samples):
            z_b = np.dot(b_weighting[i], ic_i_val)
            u_weights = np.exp(-0.5 * (z_b - np.min(z_b)))
            z_bs[i] = z_b
            weights[i] = u_weights / np.sum(u_weights)

        weights = weights.mean(0)
        ses = pd.Series(z_bs.std(0))

    elif method == 'pseudo-BMA':
        min_ic = ics.iloc[0][ic]
        Z = np.exp(-0.5 * (ics[ic] - min_ic))
        weights = Z / np.sum(Z)
        ses = ics[ic_se]

    if np.any(weights):
        min_ic_i_val = ics[ic_i].iloc[0]
        for i in ics.index:
            res = ics.loc[i]
            diff = res[ic_i] - min_ic_i_val
            d_ic = np.sum(diff)
            d_se = np.sqrt(len(diff) * np.var(diff))
            se = ses.loc[i]
            weight = weights[i]
            df_comp.at[i] = (round(res[ic], round_to),
                             round(res[p_ic], round_to),
                             round(d_ic, round_to),
                             round(weight, round_to),
                             round(se, round_to),
                             round(d_se, round_to),
                             res['warning'])

        return df_comp.sort_values(by=ic)


def _ic_matrix(ics, ic_i):
    """
    Store the previously computed pointwise predictive accuracy values (ics) in a 2D matrix array.
    """
    N = len(ics[ic_i].iloc[0])
    K = len(ics)
    ic_i_val = np.zeros((N, K))

    for i in ics.index:
        ic = ics.loc[i][ic_i]
        if len(ic) != N:
            raise ValueError('The number of observations should be the same '
                             'across all models')
        else:
            ic_i_val[:, i] = ic

    return N, K, ic_i_val


def hpd(x, alpha=0.05, transform=lambda x: x, circular=False):
    """
    Calculate highest posterior density (HPD) of array for given alpha.

    The HPD is the minimum width Bayesian credible interval (BCI). This implementation works only
    for unimodal distributions.

    Parameters
    ----------
    x : Numpy array
        An array containing posterior samples
    alpha : float, optional
        Desired probability of type I error (defaults to 0.05)
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
    # Make a copy of trace
    x = transform(x.copy())
    n = len(x)
    cred_mass = 1.0 - alpha

    if circular:
        mean = circmean(x, high=np.pi, low=-np.pi)
        x = x - mean
        x = np.arctan2(np.sin(x), np.cos(x))

    x = np.sort(x)
    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]

    if circular:
        hdi_min = hdi_min + mean
        hdi_max = hdi_max + mean
        hdi_min = np.arctan2(np.sin(hdi_min), np.cos(hdi_min))
        hdi_max = np.arctan2(np.sin(hdi_max), np.cos(hdi_max))

    return hdi_min, hdi_max


def _hpd_df(x, alpha):
    cnames = ['hpd_{0:g}'.format(100 * alpha / 2),
              'hpd_{0:g}'.format(100 * (1 - alpha / 2))]
    return pd.DataFrame(hpd(x, alpha), columns=cnames)


def loo(trace, model, pointwise=False, reff=None):
    """
    Pareto-smoothed importance sampling leave-one-out cross-validation

    Calculates leave-one-out (LOO) cross-validation for out of sample predictive model fit,
    following Vehtari et al. (2015). Cross-validation is computed using Pareto-smoothed
    importance sampling (PSIS).

    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    pointwise: bool, optional
        if True the pointwise predictive accuracy will be returned. Defaults to False
    reff : float, optional
        relative MCMC efficiency, `effective_n / n` i.e. number of effective samples divided by
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

    if reff is None:
        df = trace_to_dataframe(trace, combined=False)
        nchains = df.columns.value_counts()[0]
        if nchains == 1:
            reff = 1.
        else:
            eff_ave = effective_n(df).mean()
            samples = len(df) * nchains
            reff = eff_ave / samples

    log_py = log_post_trace(trace, model)

    lw, ks = psislw(-log_py, reff)
    lw += log_py

    warn_mg = 0
    if np.any(ks > 0.7):
        warnings.warn("""Estimated shape parameter of Pareto distribution is greater than 0.7 for
        one or more samples. You should consider using a more robust model, this is because
        importance sampling is less likely to work well if the marginal posterior and LOO posterior
        are very different. This is more likely to happen with a non-robust model and highly
        influential observations.""")
        warn_mg = 1

    loo_lppd_i = - 2 * logsumexp(lw, axis=0)
    loo_lppd = loo_lppd_i.sum()
    loo_lppd_se = (len(loo_lppd_i) * np.var(loo_lppd_i)) ** 0.5
    lppd = np.sum(logsumexp(log_py, axis=0, b=1. / log_py.shape[0]))
    p_loo = lppd + (0.5 * loo_lppd)

    if pointwise:
        if np.equal(loo_lppd, loo_lppd_i).all():
            warnings.warn("""The point-wise LOO is the same with the sum LOO, please double check
            the Observed RV in your model to make sure it returns element-wise logp.
            """)
        return pd.DataFrame([[loo_lppd, loo_lppd_se, p_loo, warn_mg, loo_lppd_i]],
                            columns=['loo', 'loo_se', 'p_loo', 'warning', 'loo_i'])
    else:
        return pd.DataFrame([[loo_lppd, loo_lppd_se, p_loo, warn_mg, loo_lppd_i]],
                            columns=['loo', 'loo_se', 'p_loo', 'warning', 'loo_i'])


def psislw(lw, reff=1.):
    """
    Pareto smoothed importance sampling (PSIS).

    Parameters
    ----------
    lw : array
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
    n, m = lw.shape

    lw_out = np.copy(lw, order='F')
    kss = np.empty(m)

    # precalculate constants
    cutoff_ind = - int(np.ceil(min(n / 5., 3 * (n / reff) ** 0.5))) - 1
    cutoffmin = np.log(np.finfo(float).tiny)  #pylint: disable=no-member
    k_min = 1. / 3

    # loop over sets of log weights
    for i, x in enumerate(lw_out.T):
        # improve numerical accuracy
        x -= np.max(x)
        # sort the array
        x_sort_ind = np.argsort(x)
        # divide log weights into body and right tail
        xcutoff = max(x[x_sort_ind[cutoff_ind]], cutoffmin)

        expxcutoff = np.exp(xcutoff)
        tailinds, = np.where(x > xcutoff)
        x2 = x[tailinds]
        n2 = len(x2)
        if n2 <= 4:
            # not enough tail samples for gpdfit
            k = np.inf
        else:
            # order of tail samples
            x2si = np.argsort(x2)
            # fit generalized Pareto distribution to the right tail samples
            x2 = np.exp(x2) - expxcutoff
            k, sigma = _gpdfit(x2[x2si])

        if k >= k_min and not np.isinf(k):
            # no smoothing if short tail or GPD fit failed
            # compute ordered statistic for the fit
            sti = np.arange(0.5, n2) / n2
            qq = _gpinv(sti, k, sigma)
            qq = np.log(qq + expxcutoff)
            # place the smoothed tail into the output array
            x[tailinds[x2si]] = qq
            # truncate smoothed values to the largest raw weight 0
            x[x > 0] = 0
        # renormalize weights
        x -= logsumexp(x)
        # store tail index k
        kss[i] = k

    return lw_out, kss


def _gpdfit(x):
    """
    Estimate the parameters for the Generalized Pareto Distribution (GPD)
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
    n = len(x)
    m = 30 + int(n**0.5)

    bs = 1 - np.sqrt(m / (np.arange(1, m + 1, dtype=float) - 0.5))
    bs /= prior_bs * x[int(n/4 + 0.5) - 1]
    bs += 1 / x[-1]

    ks = np.log1p(-bs[:, None] * x).mean(axis=1)
    L = n * (np.log(-(bs / ks)) - ks - 1)
    w = 1 / np.exp(L - L[:, None]).sum(axis=1)

    # remove negligible weights
    dii = w >= 10 * np.finfo(float).eps
    if not np.all(dii):
        w = w[dii]
        bs = bs[dii]
    # normalise w
    w /= w.sum()

    # posterior mean for b
    b = np.sum(bs * w)
    # estimate for k
    k = np.log1p(-b * x).mean()
    # add prior for k
    k = (n * k + prior_k * 0.5) / (n + prior_k)
    sigma = - k / b

    return k, sigma


def _gpinv(p, k, sigma):
    """Inverse Generalized Pareto distribution function"""
    x = np.full_like(p, np.nan)
    if sigma <= 0:
        return x
    ok = (p > 0) & (p < 1)
    if np.all(ok):
        if np.abs(k) < np.finfo(float).eps:
            x = - np.log1p(-p)
        else:
            x = np.expm1(-k * np.log1p(-p)) / k
        x *= sigma
    else:
        if np.abs(k) < np.finfo(float).eps:
            x[ok] = - np.log1p(-p[ok])
        else:
            x[ok] = np.expm1(-k * np.log1p(-p[ok])) / k
        x *= sigma
        x[p == 0] = 0
        if k >= 0:
            x[p == 1] = np.inf
        else:
            x[p == 1] = - sigma / k

    return x


def r2_score(y_true, y_pred, round_to=2):
    """
    R-squared for Bayesian regression models. Only valid for linear models.
    http://www.stat.columbia.edu/%7Egelman/research/unpublished/bayes_R2.pdf

    Parameters
    ----------
    y_true: : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    round_to : int
        Number of decimals used to round results (default 2).

    Returns
    -------
    Pandas Series with the following indices:
    R2_median: median of the Bayesian R2
    R2_mean: mean of the Bayesian R2
    R2_std: standard deviation of the Bayesian R2
    """
    dimension = None
    if y_true.ndim > 1:
        dimension = 1

    var_y_est = np.var(y_pred, axis=dimension)
    var_e = np.var(y_true - y_pred, axis=dimension)

    r2 = var_y_est / (var_y_est + var_e)
    return pd.Series([np.median(r2), np.mean(r2), np.std(r2)],
                     index=['r2_median', 'r2_mean', 'r2_std']).round(decimals=round_to)


def summary(trace, varnames=None, round_to=2, transform=lambda x: x, circ_varnames=None,
            stat_funcs=None, extend=False, alpha=0.05, skip_first=0, batches=None):
    R"""
    Create a data frame with summary statistics.

    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list
        Names of variables to include in summary
    round_to : int
        Controls formatting for floating point numbers. Default 2.
    transform : callable
        Function to transform data (defaults to identity)
    circ_varnames : list
        Names of circular variables to include in summary
    stat_funcs : None or list
        A list of functions used to calculate statistics. By default, the mean, standard deviation,
        simulation standard error, and highest posterior density intervals are included.

        The functions will be given one argument, the samples for a variable as a 2-D array,
        where the first axis corresponds to sampling iterations and the second axis represents the
        flattened variable (e.g., x__0, x__1,...). Each function should return either

        1) A `pandas.Series` instance containing the result of calculating the statistic along the
           first axis. The name attribute will be taken as the name of the statistic.
        2) A `pandas.DataFrame` where each column contains the result of calculating the statistic
           along the first axis. The column names will be taken as the names of the statistics.
    extend : boolean
        If True, use the statistics returned by `stat_funcs` in addition to, rather than in place
        of, the default statistics. This is only meaningful when `stat_funcs` is not None.
    include_transformed : bool
        Flag for reporting automatically transformed variables in addition to original variables
        (defaults to False).
    alpha : float
        The alpha level for generating posterior intervals. Defaults to 0.05. This is only
        meaningful when `stat_funcs` is None.
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    batches : None or int
        Batch size for calculating standard deviation for non-independent samples. Defaults to the
        smaller of 100 or the number of samples. This is only meaningful when `stat_funcs` is None.

    Returns
    -------
    `pandas.DataFrame` with summary statistics for each variable Defaults one are: `mean`, `sd`,
    `mc_error`, `hpd_2.5`, `hpd_97.5`, `n_eff` and `Rhat`. Last two are only computed for traces
    with 2 or more chains.

    Examples
    --------

    .. code:: ipython

        >>> az.summary(trace, ['mu'])
                   mean        sd  mc_error     hpd_5    hpd_95  n_eff      Rhat
        mu__0  0.106897  0.066473  0.001818 -0.020612  0.231626  487.0   1.00001
        mu__1 -0.046597  0.067513  0.002048 -0.174753  0.081924  379.0   1.00203

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
                     sd         5        50        95
        mu__0  0.066473  0.000312  0.105039  0.214242
        mu__1  0.067513 -0.159097 -0.045637  0.062912
    """
    trace = trace_to_dataframe(trace, combined=False)[skip_first:]
    varnames = get_varnames(trace, varnames)

    if batches is None:
        batches = min([100, len(trace)])

    if circ_varnames is None:
        circ_varnames = []
    else:
        circ_varnames = get_varnames(trace, circ_varnames)

    cnames = ['hpd_{0:g}'.format(100 * alpha / 2),
              'hpd_{0:g}'.format(100 * (1 - alpha / 2))]

    funcs = [lambda x: pd.Series(np.mean(x, 0), name='mean').round(round_to),
             lambda x: pd.Series(np.std(x, 0), name='sd').round(round_to),
             lambda x: pd.Series(_mc_error(x, batches).round(round_to), name='mc_error'),
             lambda x: pd.DataFrame([hpd(x, alpha)], columns=cnames).round(round_to)]

    circ_funcs = [lambda x: pd.Series(circmean(x, high=np.pi, low=-np.pi, axis=0),
                                      name='mean').round(round_to),
                  lambda x: pd.Series(circstd(x, high=np.pi, low=-np.pi, axis=0),
                                      name='sd').round(round_to),
                  lambda x: pd.Series(_mc_error(x, batches, circular=True).round(
                      round_to), name='mc_error'),
                  lambda x: pd.DataFrame([hpd(x, alpha, circular=True)],
                                         columns=cnames).round(round_to)]

    if stat_funcs is not None:
        if extend:
            funcs = funcs + stat_funcs
        else:
            funcs = stat_funcs

    var_dfs = []
    for var in varnames:
        vals = transform(np.ravel(trace[var].values))
        if var in circ_varnames:
            var_df = pd.concat([f(vals) for f in circ_funcs], axis=1)
        else:
            var_df = pd.concat([f(vals) for f in funcs], axis=1)
        var_df.index = [var]
        var_dfs.append(var_df)
    dforg = pd.concat(var_dfs, axis=0)

    if (stat_funcs is not None) and (not extend):
        return dforg
    elif trace.columns.value_counts()[0] < 2:
        return dforg
    else:
        n_eff = effective_n(trace, varnames=varnames, round_to=round_to)
        rhat = gelman_rubin(trace, varnames=varnames, round_to=round_to)
        return pd.concat([dforg, n_eff, rhat], axis=1, join_axes=[dforg.index])


def _mc_error(x, batches=5, circular=False):
    """
    Calculates the simulation standard error, accounting for non-independent
    samples. The trace is divided into batches, and the standard deviation of
    the batch means is calculated.

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

        return np.reshape([mc_error(t, batches) for t in trace], dims[1:])

    else:
        if batches == 1:
            if circular:
                std = circstd(x, high=np.pi, low=-np.pi)
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
            means = circmean(batched_traces, high=np.pi, low=-np.pi, axis=1)
            std = circstd(means, high=np.pi, low=-np.pi)
        else:
            means = np.mean(batched_traces, 1)
            std = np.std(means)

        return std / np.sqrt(batches)


def waic(trace, model, pointwise=False):
    """
    Calculate the widely available information criterion, its standard error and the effective
    number of parameters of the samples in trace from model.
    Read more theory here - in a paper by some of the leading authorities on model selection
    dx.doi.org/10.1111/1467-9868.00353

    Parameters
    ----------
    trace : result of MCMC run
    model : Probabilistic Model
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

    log_py = log_post_trace(trace, model)

    lppd_i = logsumexp(log_py, axis=0, b=1.0 / log_py.shape[0])

    vars_lpd = np.var(log_py, axis=0)
    warn_mg = 0
    if np.any(vars_lpd > 0.4):
        warnings.warn("""For one or more samples the posterior variance of the log predictive
        densities exceeds 0.4. This could be indication of WAIC starting to fail see
        http://arxiv.org/abs/1507.04544 for details
        """)
        warn_mg = 1

    waic_i = - 2 * (lppd_i - vars_lpd)
    waic_se = (len(waic_i) * np.var(waic_i))**0.5
    waic_sum = np.sum(waic_i)
    p_waic = np.sum(vars_lpd)

    if pointwise:
        if np.equal(waic_sum, waic_i).all():
            warnings.warn("""The point-wise WAIC is the same with the sum WAIC, please double check
            the Observed RV in your model to make sure it returns element-wise logp.
            """)
        return pd.DataFrame([[waic_sum, waic_se, p_waic, warn_mg, waic_i]],
                            columns=['waic', 'waic_se', 'p_waic', 'warning', 'waic_i'])
    else:
        return pd.DataFrame([[waic_sum, waic_se, p_waic, warn_mg, ]],
                            columns=['waic', 'waic_se', 'p_waic', 'warning'])
