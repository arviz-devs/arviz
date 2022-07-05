"""Marginal likelihood estimation via bridge sampling."""
import warnings
import dataclasses

import scipy.stats as st
from numpy import dot
from scipy.linalg import cholesky
import numpy as np
from scipy.linalg import lstsq
from statsmodels.tsa.ar_model import AR

from ..data import InferenceData
from ..stats.diagnostics import ess
from ..data.utils import extract_dataset

__all__ = ["log_marginal_likelihood_bridgesampling", "bridgesampling_rel_mse_est"]


def log_marginal_likelihood_bridgesampling(idata, logp, transformation_dict=None, maxiter=1000):
    """
    Log marginal likelihood estimated via bridgesampling.

    Parameters
    ----------
    idata: InferenceData
        :class:`arviz.InferenceData` object. Must have a posterior group
    logp: callable
        Unnormalized posterior log probability function.
        (E.g. model.logp_array for model, a pymc model)
    transformation_dict: dict
      Keys are (str) names of model variables in idata , values are each one's
      associated transformation - as a function that (elementwise) transforms an
      array, which should map to R for best results fitting a multivariate
      normal proposal distribution. While each (non-observed) variable must have
      a transformation in the dict, a transformation can be the identity.
    maxiter: int
      Maximum number of iterations in the iterative scheme in bridge sampling

    Returns
    -------
    log_marginal_likelihood: float
      Estimated log marginal likelihood (estimation method: bridge sampling)
    bridge_sampling_stats: BridgeSamplingStats
      Information, such as number of iterations, that could be useful for
      futher diagnostics. See documentation for BridgeSamplingStats

    References
    ----------
    [1] Gronau, Quentin F., et al. "A tutorial on bridge sampling."
    Journal of mathematical psychology 81 (2017): 80-97.
    [2] Meng, Xiao-Li, and Wing Hung Wong. "Simulating ratios of normalizing
    constants via a simple identity: a theoretical exploration." Statistica
    Sinica (1996): 831-860.

    Examples
    --------
    Estimating the log marginal likelihood for a PyMC model

    .. ipython::

        In [1]: n, k = 10, 2
           ...: with pm.Model() as model1:
           ...:   p = pm.Beta('p', alpha=1., beta=1.)
           ...:   obs = pm.Binomial('obs', p=p, n=n, observed=k)
           ...:   trace1 = pm.sample(return_inferencedata=True)
           ...:
           ...: logp = model1.logp_array
           ...: transformation_dict = {var_name: getattr(getattr(model1, var_name),
           ...:             'transformation', lambda x:x)
           ...:             for var_name in model1.named_vars
           ...:             if not var_name.endswith('__')}
           ...: log_marg_lik, stats = log_marginal_likelihood_bridgesampling(
           ...:                         trace1,
           ...:                         logp,
           ...:                         transformation_dict,
           ...:                         maxiter=1000)

    """
    r_initial, tol1, tol2 = 0.5, 1e-12, 1e-4

    # check idata input
    if not isinstance(idata, InferenceData):
        raise ValueError("idata must be of type InferenceData")
    if not "posterior" in idata.groups():
        raise ValueError("idata must have a posterior group")

    # variable names as a list of strings
    free_RV_names = list(idata.posterior.keys())  # pylint: disable=invalid-name

    if transformation_dict is None:
        warnings.warn(
            "If transformation_dict is not provided, untransformed "
            "variables will be used for fitting a multivariate normal proposal "
            "distribution. This may result in poor performance e.g if free model "
            "variables are bounded, e.g."
        )
        transformation_dict = {var_name: (lambda x: x) for var_name in free_RV_names}

    # combine chains and draws
    posterior_combined_draws_and_chains = extract_dataset(
        idata["posterior"]
    )  # combine chains and draws

    # create dictionary of the dimensions required for each variable
    var_dims = {}
    for var_name in free_RV_names:
        if len(posterior_combined_draws_and_chains[var_name].shape) < 2:
            var_dims[var_name] = 1
        else:
            var_dims[var_name] = posterior_combined_draws_and_chains[var_name].shape[0]

    # Split the samples into two parts
    # Use the first 50% for fiting the proposal distribution and the second 50%
    # in the iterative scheme.
    len_trace = len(idata["posterior"]["draw"])
    nchain = len(idata["posterior"]["chain"])

    # Keeping variable names N_1 and N_2 to match Gronau et al. (2017) [1]
    N_1 = (len_trace // 2) * nchain  # pylint: disable=invalid-name
    N_2 = len_trace * nchain - N_1  # pylint: disable=invalid-name

    neff_list = {}  # effective sample size, a dict of ess for each var

    arraysz = sum(var_dims.values())
    samples_4_fit = np.zeros((arraysz, N_1))
    samples_4_iter = np.zeros((arraysz, N_2))

    var_idx = 0
    for var_name in free_RV_names:
        # Transform the samples for proposal dist
        samples_4_fit[var_idx : var_idx + var_dims[var_name], :] = transformation_dict[var_name](
            posterior_combined_draws_and_chains[var_name].values[..., :N_1]
        )

        # for iterative procedure (apply same transformation)
        iter_samples_tmp = transformation_dict[var_name](
            posterior_combined_draws_and_chains[var_name].values[..., N_1:]
        )
        samples_4_iter[var_idx : var_idx + var_dims[var_name], :] = iter_samples_tmp

        var_idx += var_dims[var_name]
        # effective sample size of samples_4_iter, scalar
        neff_list.update({var_name: ess(iter_samples_tmp)})

    # # median effective sample size (scalar)
    neff = np.median(list(neff_list.values()))

    # get mean & covariance matrix and generate samples from proposal
    m = np.mean(samples_4_fit, axis=1)
    proposal_cov = np.cov(samples_4_fit)
    lower_chol = cholesky(proposal_cov, lower=True)

    # Draw N_2 samples from the proposal distribution
    gen_samples = m[:, None] + dot(lower_chol, st.norm.rvs(0, 1, size=samples_4_iter.shape))

    # Evaluate proposal distribution for posterior & generated samples
    q12 = st.multivariate_normal.logpdf(samples_4_iter.T, m, proposal_cov)
    q22 = st.multivariate_normal.logpdf(gen_samples.T, m, proposal_cov)

    # Evaluate unnormalized posterior for posterior & generated samples
    q11 = np.asarray([logp(point) for point in samples_4_iter.T])
    q21 = np.asarray([logp(point) for point in gen_samples.T])

    # Iterative scheme as proposed in Meng and Wong (1996) [2] to estimate
    # the marginal likelihood
    def iterative_scheme(q11, q12, q21, q22, r_initial, neff, tol, maxiter, criterion):
        log_l1 = q11 - q12
        log_l2 = q21 - q22
        lstar = np.median(log_l1)  # To increase numerical stability,
        # subtracting the median of log_l1 from log_l1 & log_l2 later

        # Keeping variable names s1, s2, r to match Gronau et al. (2017) [1]
        s1 = neff / (neff + N_2)  # pylint: disable=invalid-name
        s2 = N_2 / (neff + N_2)  # pylint: disable=invalid-name
        r = r_initial  # pylint: disable=invalid-name
        r_vals = [r]
        logml = np.log(r) + lstar
        criterion_val = 1 + tol

        i = 0
        while (i <= maxiter) & (criterion_val > tol):
            rold = r
            logmlold = logml
            numi = np.exp(log_l2 - lstar) / (s1 * np.exp(log_l2 - lstar) + s2 * r)
            deni = 1 / (s1 * np.exp(log_l1 - lstar) + s2 * r)
            if np.sum(~np.isfinite(numi)) + np.sum(~np.isfinite(deni)) > 0:
                warnings.warn(
                    """Infinite value in iterative scheme, returning NaN.
                Try rerunning with more samples."""
                )
            # Keeping variable name r to match Gronau et al. (2017) [1]
            r = (N_1 / N_2) * np.sum(numi) / np.sum(deni)  # pylint: disable=invalid-name
            r_vals.append(r)
            logml = np.log(r) + lstar
            i += 1
            if criterion == "r":
                criterion_val = np.abs((r - rold) / r)
            elif criterion == "logml":
                criterion_val = np.abs((logml - logmlold) / logml)

        if i >= maxiter:
            return dict(log_marginal_likelihood=np.NaN, niter=i, r_vals=np.asarray(r_vals))
        else:
            return dict(log_marginal_likelihood=logml, niter=i, r_vals=np.asarray(r_vals))

    # Run iterative scheme:
    tmp = iterative_scheme(q11, q12, q21, q22, r_initial, neff, tol1, maxiter, "r")

    if ~np.isfinite(tmp["log_marginal_likelihood"]):
        warnings.warn(
            """logml could not be estimated within maxiter, rerunning with
                      adjusted starting value. Estimate might be more variable than usual."""
        )
        # use geometric mean as starting value
        r0_2 = np.sqrt(tmp["r_vals"][-2] * tmp["r_vals"][-1])
        tmp = iterative_scheme(q11, q12, q21, q22, r0_2, neff, tol2, maxiter, "logml")

    log_marginal_likelihood = tmp["log_marginal_likelihood"]
    bridge_sampling_stats = BridgeSamplingStats(
        niter=tmp["niter"], method="normal", q11=q11, q12=q12, q21=q21, q22=q22
    )

    return log_marginal_likelihood, bridge_sampling_stats


@dataclasses.dataclass
class BridgeSamplingStats:
    """
    Quantities for bridge sampling diagnostics.

    Attributes
    ----------
    niter: int
        number of iterations in the bridgesampling iterative scheme
    method: str
        E.g. "normal" when multivariate normal distribution is fit
    q11: array-like
        Unnormalized log posterior density evaluated at posterior samples
    q21: array-like
        Unnormalized log posterior density evaluated at proposal samples
    q12: array-like
        Unnormalized log proposal density evaluated at posterior samples
    q22: array-like
        Unnormalized log proposal density evaluated at proposal samples
    """

    niter: int
    method: str
    q11: np.ndarray
    q21: np.ndarray
    q12: np.ndarray
    q22: np.ndarray


def _spectrum0_ar(x):
    """
    Fits an autoregressive model and gives an estimate of the spectral density at freq 0.

    Parameters
    ----------
    x: array-like
        The time series to be fit

    Returns
    -------
    spec: float
        The estimated spectral density at frequency 0

    Notes
    -----
    Port of spectrum0.ar from coda::spectrum0.ar.
    """
    z = np.arange(1, len(x) + 1)
    z = z[:, np.newaxis] ** [0, 1]
    coeffs, _, _, _ = lstsq(z, x)
    residuals = x - np.matmul(z, coeffs)

    if residuals.std() == 0:
        spec = order = 0
    else:
        ar_out = AR(x).fit(ic="aic", trend="c")
        order = ar_out.k_ar
        spec = np.var(ar_out.resid) / (1 - np.sum(ar_out.params[1:])) ** 2

    return spec, order


def bridgesampling_rel_mse_est(estimated_log_marginal_likelihood, bridge_sampling_stats):
    """
    Estimate of expected relative mean-square error E(true - est)^2 / true^2.

    Parameters
    ----------
    estimated_log_marginal_likelihood: float
      An estimate of the log marginal likelihood, obtained via bridge sampling
    bridge_sampling_stats: BridgeSamplingStats
      Includes quantities q11, q12, q21, q22

    Returns
    -------
    re2: float
      An estimate of the expected relative mean squared error of the log
      marginal likelihood. That is an *estimate* of E(true-est)^2 / true^2

    Notes
    -----
    Port of the error_measures.R in bridgesampling
    https://github.com/quentingronau/bridgesampling/blob/master/R/error_measures.R
    As proposed in:
    Frühwirth‐Schnatter, Sylvia. "Estimating marginal likelihoods for mixture and Markov
    switching models using bridge sampling techniques." The Econometrics Journal 7.1
    (2004): 143-167.
    """
    marginal_likelihood = np.exp(estimated_log_marginal_likelihood)
    g_p = np.exp(bridge_sampling_stats["q12"])
    g_g = np.exp(bridge_sampling_stats["q22"])
    prior_times_lik_p = np.exp(bridge_sampling_stats["q11"])
    prior_times_lik_g = np.exp(bridge_sampling_stats["q21"])
    p_p = prior_times_lik_p / marginal_likelihood
    p_g = prior_times_lik_g / marginal_likelihood

    # Keeping variable names N_1, N_2, s1, s2, f1 f2 to match Gronau et al. (2017) [1]
    N_1 = len(p_p)  # pylint: disable=invalid-name
    N_2 = len(g_g)  # pylint: disable=invalid-name
    s1 = N_1 / (N_1 + N_2)  # pylint: disable=invalid-name
    s2 = N_2 / (N_1 + N_2)  # pylint: disable=invalid-name

    f1 = p_g / (s1 * p_g + s2 * g_g)  # pylint: disable=invalid-name
    f2 = g_p / (s1 * p_p + s2 * g_p)  # pylint: disable=invalid-name
    rho_f2, _ = _spectrum0_ar(f2)

    term1 = 1 / N_2 * np.var(f1) / np.mean(f1) ** 2
    term2 = rho_f2 / N_1 * np.var(f2) / np.mean(f2) ** 2

    re2 = term1 + term2

    return re2
