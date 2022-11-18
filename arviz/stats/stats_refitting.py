"""Stats functions that require refitting the model."""
import logging
import warnings

import numpy as np

from .stats import loo
from .stats_utils import logsumexp as _logsumexp

__all__ = ["reloo"]

_log = logging.getLogger(__name__)


def reloo(wrapper, loo_orig=None, k_thresh=0.7, scale=None, verbose=True):
    """Recalculate exact Leave-One-Out cross validation refitting where the approximation fails.

    ``az.loo`` estimates the values of Leave-One-Out (LOO) cross validation using Pareto
    Smoothed Importance Sampling (PSIS) to approximate its value. PSIS works well when
    the posterior and the posterior_i (excluding observation i from the data used to fit)
    are similar. In some cases, there are highly influential observations for which PSIS
    cannot approximate the LOO-CV, and a warning of a large Pareto shape is sent by ArviZ.
    This cases typically have a handful of bad or very bad Pareto shapes and a majority of
    good or ok shapes.

    Therefore, this may not indicate that the model is not robust enough
    nor that these observations are inherently bad, only that PSIS cannot approximate LOO-CV
    correctly. Thus, we can use PSIS for all observations where the Pareto shape is below a
    threshold and refit the model to perform exact cross validation for the handful of
    observations where PSIS cannot be used. This approach allows to properly approximate
    LOO-CV with only a handful of refits, which in most cases is still much less computationally
    expensive than exact LOO-CV, which needs one refit per observation.

    Parameters
    ----------
    wrapper: SamplingWrapper-like
        Class (preferably a subclass of ``az.SamplingWrapper``, see :ref:`wrappers_api`
        for details) implementing the methods described
        in the SamplingWrapper docs. This allows ArviZ to call **any** sampling backend
        (like PyStan or emcee) using always the same syntax.
    loo_orig : ELPDData, optional
        ELPDData instance with pointwise loo results. The pareto_k attribute will be checked
        for values above the threshold.
    k_thresh : float, optional
        Pareto shape threshold. Each pareto shape value above ``k_thresh`` will trigger
        a refit excluding that observation.
    scale : str, optional
        Only taken into account when loo_orig is None. See ``az.loo`` for valid options.

    Returns
    -------
    ELPDData
        ELPDData instance containing the PSIS approximation where possible and the exact
        LOO-CV result where PSIS failed. The Pareto shape of the observations where exact
        LOO-CV was performed is artificially set to 0, but as PSIS is not performed, it
        should be ignored.

    Notes
    -----
    It is strongly recommended to first compute ``az.loo`` on the inference results to
    confirm that the number of values above the threshold is small enough. Otherwise,
    prohibitive computation time may be needed to perform all required refits.

    As an extreme case, artificially assigning all ``pareto_k`` values to something
    larger than the threshold would make ``reloo`` perform the whole exact LOO-CV.
    This is not generally recommended
    nor intended, however, if needed, this function can be used to achieve the result.

    Warnings
    --------
    Sampling wrappers are an experimental feature in a very early stage. Please use them
    with caution.
    """
    required_methods = ("sel_observations", "sample", "get_inference_data", "log_likelihood__i")
    not_implemented = wrapper.check_implemented_methods(required_methods)
    if not_implemented:
        raise TypeError(
            "Passed wrapper instance does not implement all methods required for reloo "
            f"to work. Check the documentation of SamplingWrapper. {not_implemented} must be "
            "implemented and were not found."
        )
    if loo_orig is None:
        loo_orig = loo(wrapper.idata_orig, pointwise=True, scale=scale)
    loo_refitted = loo_orig.copy()
    khats = loo_refitted.pareto_k
    loo_i = loo_refitted.loo_i
    scale = loo_orig.scale

    if scale.lower() == "deviance":
        scale_value = -2
    elif scale.lower() == "log":
        scale_value = 1
    elif scale.lower() == "negative_log":
        scale_value = -1
    lppd_orig = loo_orig.p_loo + loo_orig.elpd_loo / scale_value
    n_data_points = loo_orig.n_data_points

    if verbose:
        warnings.warn("reloo is an experimental and untested feature", UserWarning)

    if np.any(khats > k_thresh):
        for idx in np.argwhere(khats.values > k_thresh):
            if verbose:
                _log.info("Refitting model excluding observation %d", idx)
            new_obs, excluded_obs = wrapper.sel_observations(idx)
            fit = wrapper.sample(new_obs)
            idata_idx = wrapper.get_inference_data(fit)
            log_like_idx = wrapper.log_likelihood__i(excluded_obs, idata_idx).values.flatten()
            loo_lppd_idx = scale_value * _logsumexp(log_like_idx, b_inv=len(log_like_idx))
            khats[idx] = 0
            loo_i[idx] = loo_lppd_idx
        loo_refitted.elpd_loo = loo_i.values.sum()
        loo_refitted.se = (n_data_points * np.var(loo_i.values)) ** 0.5
        loo_refitted.p_loo = lppd_orig - loo_refitted.elpd_loo / scale_value
        return loo_refitted
    else:
        _log.info("No problematic observations")
        return loo_orig
