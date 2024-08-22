"""Functions for evaluating ECDFs and their confidence bands."""

import math
from typing import Any, Callable, Optional, Tuple
import warnings

import numpy as np
from scipy.stats import uniform, binom
from scipy.optimize import minimize_scalar

try:
    from numba import jit, vectorize
except ImportError:

    def jit(*args, **kwargs):  # pylint: disable=unused-argument
        return lambda f: f

    def vectorize(*args, **kwargs):  # pylint: disable=unused-argument
        return lambda f: f


from ..utils import Numba


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
    method="optimized",
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
        - "optimized": Use optimization to estimate a simultaneous confidence band.
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
    elif method == "optimized":
        prob_pointwise = _optimize_simultaneous_ecdf_band_probability(
            ndraws, eval_points, cdf_at_eval_points, prob=prob, **kwargs
        )
    elif method == "simulated":
        prob_pointwise = _simulate_simultaneous_ecdf_band_probability(
            ndraws, eval_points, cdf_at_eval_points, prob=prob, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown method {method}. Valid options are 'pointwise', 'optimized', or 'simulated'."
        )

    prob_lower, prob_upper = _get_pointwise_confidence_band(
        prob_pointwise, ndraws, cdf_at_eval_points
    )

    return prob_lower, prob_upper


def _update_ecdf_band_interior_probabilities(
    prob_left: np.ndarray,
    interval_left: np.ndarray,
    interval_right: np.ndarray,
    p: float,
    ndraws: int,
) -> np.ndarray:
    """Update the probability that an ECDF has been within the envelope including at the current
    point.

    Arguments
    ---------
    prob_left : np.ndarray
        For each point in the interior at the previous point, the joint probability that it and all
        points before are in the interior.
    interval_left : np.ndarray
        The set of points in the interior at the previous point.
    interval_right : np.ndarray
        The set of points in the interior at the current point.
    p : float
        The probability of any given point found between the previous point and the current one.
    ndraws : int
        Number of draws in the original dataset.

    Returns
    -------
    prob_right : np.ndarray
        For each point in the interior at the current point, the joint probability that it and all
        previous points are in the interior.
    """
    interval_left = interval_left[:, np.newaxis]
    prob_conditional = binom.pmf(interval_right, ndraws - interval_left, p, loc=interval_left)
    prob_right = prob_left.dot(prob_conditional)
    return prob_right


@vectorize(["float64(int64, int64, float64, int64)"])
def _binom_pmf(k, n, p, loc):
    k -= loc
    if k < 0 or k > n:
        return 0.0
    if p == 0:
        return 1.0 if k == 0 else 0.0
    if p == 1:
        return 1.0 if k == n else 0.0
    if k == 0:
        return (1 - p) ** n
    if k == n:
        return p**n
    lbinom = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return np.exp(lbinom + k * np.log(p) + (n - k) * np.log1p(-p))


@jit(nopython=True)
def _update_ecdf_band_interior_probabilities_numba(
    prob_left: np.ndarray,
    interval_left: np.ndarray,
    interval_right: np.ndarray,
    p: float,
    ndraws: int,
) -> np.ndarray:
    interval_left = interval_left[:, np.newaxis]
    prob_conditional = _binom_pmf(interval_right, ndraws - interval_left, p, interval_left)
    prob_right = prob_left.dot(prob_conditional)
    return prob_right


def _ecdf_band_interior_probability(prob_between_points, ndraws, lower_count, upper_count):
    interval_left = np.arange(1)
    prob_interior = np.ones(1)
    for i in range(prob_between_points.shape[0]):
        interval_right = np.arange(lower_count[i], upper_count[i])
        prob_interior = _update_ecdf_band_interior_probabilities(
            prob_interior, interval_left, interval_right, prob_between_points[i], ndraws
        )
        interval_left = interval_right
    return prob_interior.sum()


@jit(nopython=True)
def _ecdf_band_interior_probability_numba(prob_between_points, ndraws, lower_count, upper_count):
    interval_left = np.arange(1)
    prob_interior = np.ones(1)
    for i in range(prob_between_points.shape[0]):
        interval_right = np.arange(lower_count[i], upper_count[i])
        prob_interior = _update_ecdf_band_interior_probabilities_numba(
            prob_interior, interval_left, interval_right, prob_between_points[i], ndraws
        )
        interval_left = interval_right
    return prob_interior.sum()


def _ecdf_band_optimization_objective(
    prob_pointwise: float,
    cdf_at_eval_points: np.ndarray,
    ndraws: int,
    prob_target: float,
) -> float:
    """Objective function for optimizing the simultaneous confidence band probability."""
    lower, upper = _get_pointwise_confidence_band(prob_pointwise, ndraws, cdf_at_eval_points)
    lower_count = (lower * ndraws).astype(int)
    upper_count = (upper * ndraws).astype(int) + 1
    cdf_with_zero = np.insert(cdf_at_eval_points[:-1], 0, 0)
    prob_between_points = (cdf_at_eval_points - cdf_with_zero) / (1 - cdf_with_zero)
    if Numba.numba_flag:
        prob_interior = _ecdf_band_interior_probability_numba(
            prob_between_points, ndraws, lower_count, upper_count
        )
    else:
        prob_interior = _ecdf_band_interior_probability(
            prob_between_points, ndraws, lower_count, upper_count
        )
    return abs(prob_interior - prob_target)


def _optimize_simultaneous_ecdf_band_probability(
    ndraws: int,
    eval_points: np.ndarray,  # pylint: disable=unused-argument
    cdf_at_eval_points: np.ndarray,
    prob: float = 0.95,
    **kwargs,  # pylint: disable=unused-argument
):
    """Estimate probability for simultaneous confidence band using optimization.

    This function simulates the pointwise probability needed to construct pointwise confidence bands
    that form a `prob`-level confidence envelope for the ECDF of a sample.
    """
    cdf_at_eval_points = np.unique(cdf_at_eval_points)
    objective = lambda p: _ecdf_band_optimization_objective(p, cdf_at_eval_points, ndraws, prob)
    prob_pointwise = minimize_scalar(objective, bounds=(prob, 1), method="bounded").x
    return prob_pointwise


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
