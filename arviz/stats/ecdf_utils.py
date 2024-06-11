"""Functions for evaluating ECDFs and their confidence bands."""

from typing import Any, Callable, Optional, Tuple
import warnings

import numpy as np
from scipy.stats import uniform, binom


def compute_ecdf(sample: np.ndarray, eval_points: np.ndarray) -> np.ndarray:
    """Compute ECDF of the sorted `sample` at the evaluation points."""
    return np.searchsorted(sample, eval_points, side="right") / len(sample)


def _get_ecdf_points(
    sample: np.ndarray, eval_points: np.ndarray, difference: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the coordinates for the ecdf points using compute_ecdf."""
    x = eval_points
    y = compute_ecdf(sample, eval_points)

    if not difference and y[0] > 0:
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, 0)
    return x, y


def _call_rvs(rvs, ndraws, random_state):
    if random_state is None:
        return rvs(ndraws)
    else:
        return rvs(ndraws, random_state=random_state)


def _simulate_ecdf(
    ndraws: int,
    eval_points: np.ndarray,
    rvs: Callable[[int, Optional[Any]], np.ndarray],
    random_state: Optional[Any] = None,
) -> np.ndarray:
    """Simulate ECDF at the `eval_points` using the given random variable sampler"""
    sample = _call_rvs(rvs, ndraws, random_state)
    sample.sort()
    return compute_ecdf(sample, eval_points)


def _fit_pointwise_band_probability(
    ndraws: int,
    ecdf_at_eval_points: np.ndarray,
    cdf_at_eval_points: np.ndarray,
) -> float:
    """Compute the smallest marginal probability of a pointwise confidence band that
    contains the ECDF."""
    ecdf_scaled = (ndraws * ecdf_at_eval_points).astype(int)
    prob_lower_tail = np.amin(binom.cdf(ecdf_scaled, ndraws, cdf_at_eval_points))
    prob_upper_tail = np.amin(binom.sf(ecdf_scaled - 1, ndraws, cdf_at_eval_points))
    prob_pointwise = 1 - 2 * min(prob_lower_tail, prob_upper_tail)
    return prob_pointwise


def _get_pointwise_confidence_band(
    prob: float, ndraws: int, cdf_at_eval_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the `prob`-level pointwise confidence band."""
    count_lower, count_upper = binom.interval(prob, ndraws, cdf_at_eval_points)
    prob_lower = count_lower / ndraws
    prob_upper = count_upper / ndraws
    return prob_lower, prob_upper


def ecdf_confidence_band(
    ndraws: int,
    eval_points: np.ndarray,
    cdf_at_eval_points: np.ndarray,
    prob: float = 0.95,
    method="simulated",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the `prob`-level confidence band for the ECDF.

    Arguments
    ---------
    ndraws : int
        Number of samples in the original dataset.
    eval_points : np.ndarray
        Points at which the ECDF is evaluated. If these are dependent on the sample
        values, simultaneous confidence bands may not be correctly calibrated.
    cdf_at_eval_points : np.ndarray
        CDF values at the evaluation points.
    prob : float, default 0.95
        The target probability that a true ECDF lies within the confidence band.
    method : string, default "simulated"
        The method used to compute the confidence band. Valid options are:
        - "pointwise": Compute the pointwise (i.e. marginal) confidence band.
        - "simulated": Use Monte Carlo simulation to estimate a simultaneous confidence band.
          `rvs` must be provided.
    rvs: callable, optional
        A function that takes an integer `ndraws` and optionally the object passed to
        `random_state` and returns an array of `ndraws` samples from the same distribution
        as the original dataset. Required if `method` is "simulated" and variable is discrete.
    num_trials : int, default 500
        The number of random ECDFs to generate for constructing simultaneous confidence bands
        (if `method` is "simulated").
    random_state : int, numpy.random.Generator or numpy.random.RandomState, optional

    Returns
    -------
    prob_lower : np.ndarray
        Lower confidence band for the ECDF at the evaluation points.
    prob_upper : np.ndarray
        Upper confidence band for the ECDF at the evaluation points.
    """
    if not 0 < prob < 1:
        raise ValueError(f"Invalid value for `prob`. Expected 0 < prob < 1, but got {prob}.")

    if method == "pointwise":
        prob_pointwise = prob
    elif method == "simulated":
        prob_pointwise = _simulate_simultaneous_ecdf_band_probability(
            ndraws, eval_points, cdf_at_eval_points, prob=prob, **kwargs
        )
    else:
        raise ValueError(f"Unknown method {method}. Valid options are 'pointwise' or 'simulated'.")

    prob_lower, prob_upper = _get_pointwise_confidence_band(
        prob_pointwise, ndraws, cdf_at_eval_points
    )

    return prob_lower, prob_upper


def _simulate_simultaneous_ecdf_band_probability(
    ndraws: int,
    eval_points: np.ndarray,
    cdf_at_eval_points: np.ndarray,
    prob: float = 0.95,
    rvs: Optional[Callable[[int, Optional[Any]], np.ndarray]] = None,
    num_trials: int = 500,
    random_state: Optional[Any] = None,
) -> float:
    """Estimate probability for simultaneous confidence band using simulation.

    This function simulates the pointwise probability needed to construct pointwise
    confidence bands that form a `prob`-level confidence envelope for the ECDF
    of a sample.
    """
    if rvs is None:
        warnings.warn(
            "Assuming variable is continuous for calibration of pointwise bands. "
            "If the variable is discrete, specify random variable sampler `rvs`.",
            UserWarning,
        )
        # if variable continuous, we can calibrate the confidence band using a uniform
        # distribution
        rvs = uniform(0, 1).rvs
        eval_points_sim = cdf_at_eval_points
    else:
        eval_points_sim = eval_points

    probs_pointwise = np.empty(num_trials)
    for i in range(num_trials):
        ecdf_at_eval_points = _simulate_ecdf(
            ndraws, eval_points_sim, rvs, random_state=random_state
        )
        prob_pointwise = _fit_pointwise_band_probability(
            ndraws, ecdf_at_eval_points, cdf_at_eval_points
        )
        probs_pointwise[i] = prob_pointwise
    return np.quantile(probs_pointwise, prob)
