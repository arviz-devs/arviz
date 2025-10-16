import pytest

import numpy as np
import scipy.stats
from ...stats.ecdf_utils import (
    compute_ecdf,
    ecdf_confidence_band,
    _get_ecdf_points,
    _simulate_ecdf,
    _get_pointwise_confidence_band,
)

try:
    import numba  # pylint: disable=unused-import

    numba_options = [True, False]  # pylint: disable=invalid-name
except ImportError:
    numba_options = [False]  # pylint: disable=invalid-name


def test_compute_ecdf():
    """Test compute_ecdf function."""
    sample = np.array([1, 2, 3, 3, 4, 5])
    eval_points = np.arange(0, 7, 0.1)
    ecdf_expected = (sample[:, None] <= eval_points).mean(axis=0)
    assert np.allclose(compute_ecdf(sample, eval_points), ecdf_expected)
    assert np.allclose(compute_ecdf(sample / 2 + 10, eval_points / 2 + 10), ecdf_expected)


@pytest.mark.parametrize("difference", [True, False])
def test_get_ecdf_points(difference):
    """Test _get_ecdf_points."""
    # if first point already outside support, no need to insert it
    sample = np.array([1, 2, 3, 3, 4, 5, 5])
    eval_points = np.arange(-1, 7, 0.1)
    x, y = _get_ecdf_points(sample, eval_points, difference)
    assert np.array_equal(x, eval_points)
    assert np.array_equal(y, compute_ecdf(sample, eval_points))

    # if first point is inside support, insert it if not in difference mode
    eval_points = np.arange(1, 6, 0.1)
    x, y = _get_ecdf_points(sample, eval_points, difference)
    assert len(x) == len(eval_points) + 1 - difference
    assert len(y) == len(eval_points) + 1 - difference

    # if not in difference mode, first point should be (eval_points[0], 0)
    if not difference:
        assert x[0] == eval_points[0]
        assert y[0] == 0
        assert np.allclose(x[1:], eval_points)
        assert np.allclose(y[1:], compute_ecdf(sample, eval_points))
        assert x[-1] == eval_points[-1]
        assert y[-1] == 1


@pytest.mark.parametrize(
    "dist", [scipy.stats.norm(3, 10), scipy.stats.binom(10, 0.5)], ids=["continuous", "discrete"]
)
@pytest.mark.parametrize("seed", [32, 87])
def test_simulate_ecdf(dist, seed):
    """Test _simulate_ecdf."""
    ndraws = 1000
    eval_points = np.arange(0, 1, 0.1)

    rvs = dist.rvs

    random_state = np.random.default_rng(seed)
    ecdf = _simulate_ecdf(ndraws, eval_points, rvs, random_state=random_state)
    random_state = np.random.default_rng(seed)
    ecdf_expected = compute_ecdf(np.sort(rvs(ndraws, random_state=random_state)), eval_points)

    assert np.allclose(ecdf, ecdf_expected)


@pytest.mark.parametrize("prob", [0.8, 0.9])
@pytest.mark.parametrize(
    "dist", [scipy.stats.norm(3, 10), scipy.stats.poisson(100)], ids=["continuous", "discrete"]
)
@pytest.mark.parametrize("ndraws", [10_000])
def test_get_pointwise_confidence_band(dist, prob, ndraws, num_trials=1_000, seed=57):
    """Test _get_pointwise_confidence_band."""
    eval_points = np.linspace(*dist.interval(0.99), 10)
    cdf_at_eval_points = dist.cdf(eval_points)

    ecdf_lower, ecdf_upper = _get_pointwise_confidence_band(prob, ndraws, cdf_at_eval_points)

    # check basic properties
    assert np.all(ecdf_lower >= 0)
    assert np.all(ecdf_upper <= 1)
    assert np.all(ecdf_lower <= ecdf_upper)

    # use simulation to estimate lower and upper bounds on pointwise probability
    in_interval = []
    random_state = np.random.default_rng(seed)
    for _ in range(num_trials):
        ecdf = _simulate_ecdf(ndraws, eval_points, dist.rvs, random_state=random_state)
        in_interval.append((ecdf_lower <= ecdf) & (ecdf < ecdf_upper))
    asymptotic_dist = scipy.stats.norm(
        np.mean(in_interval, axis=0), scipy.stats.sem(in_interval, axis=0)
    )
    prob_lower, prob_upper = asymptotic_dist.interval(0.999)

    # check target probability within all bounds
    assert np.all(prob_lower <= prob)
    assert np.all(prob <= prob_upper)


@pytest.mark.parametrize("prob", [0.8, 0.9])
@pytest.mark.parametrize(
    "dist, rvs",
    [
        (scipy.stats.norm(3, 10), scipy.stats.norm(3, 10).rvs),
        (scipy.stats.norm(3, 10), None),
        (scipy.stats.poisson(100), scipy.stats.poisson(100).rvs),
    ],
    ids=["continuous", "continuous default rvs", "discrete"],
)
@pytest.mark.parametrize("ndraws", [10_000])
@pytest.mark.parametrize("method", ["pointwise", "optimized", "simulated"])
@pytest.mark.parametrize("use_numba", numba_options)
def test_ecdf_confidence_band(
    dist, rvs, prob, ndraws, method, use_numba, num_trials=1_000, seed=57
):
    """Test test_ecdf_confidence_band."""
    if use_numba and method != "optimized":
        pytest.skip("Numba only used in optimized method")

    eval_points = np.linspace(*dist.interval(0.99), 10)
    cdf_at_eval_points = dist.cdf(eval_points)
    random_state = np.random.default_rng(seed)

    ecdf_lower, ecdf_upper = ecdf_confidence_band(
        ndraws,
        eval_points,
        cdf_at_eval_points,
        prob=prob,
        rvs=rvs,
        random_state=random_state,
        method=method,
    )

    if method == "pointwise":
        # these values tested elsewhere, we just make sure they're the same
        ecdf_lower_pointwise, ecdf_upper_pointwise = _get_pointwise_confidence_band(
            prob, ndraws, cdf_at_eval_points
        )
        assert np.array_equal(ecdf_lower, ecdf_lower_pointwise)
        assert np.array_equal(ecdf_upper, ecdf_upper_pointwise)
        return

    # check basic properties
    assert np.all(ecdf_lower >= 0)
    assert np.all(ecdf_upper <= 1)
    assert np.all(ecdf_lower <= ecdf_upper)

    # use simulation to estimate lower and upper bounds on simultaneous probability
    in_envelope = []
    random_state = np.random.default_rng(seed)
    for _ in range(num_trials):
        ecdf = _simulate_ecdf(ndraws, eval_points, dist.rvs, random_state=random_state)
        in_envelope.append(np.all(ecdf_lower <= ecdf) & np.all(ecdf < ecdf_upper))
    asymptotic_dist = scipy.stats.norm(np.mean(in_envelope), scipy.stats.sem(in_envelope))
    prob_lower, prob_upper = asymptotic_dist.interval(0.999)

    # check target probability within bounds
    assert prob_lower <= prob <= prob_upper
