import pytest

import numpy as np
import scipy.stats
from ...data import load_arviz_data
from ...stats.density_utils import (
    _prepare_cv_score_inputs,
    _compute_cv_score,
    _bw_cv,
    _bw_oversmoothed,
    _bw_scott,
    histogram,
    kde,
)


def compute_cv_score_explicit(bw, x, unbiased):
    """Explicit computation of the CV score for a 1D dataset."""
    n = len(x)
    score = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            delta = (x[i] - x[j]) / bw
            if unbiased:
                score += np.exp(-0.25 * delta**2) - np.sqrt(8) * np.exp(-0.5 * delta**2)
            else:
                score += (delta**4 - 12 * delta**2 + 12) * np.exp(-0.25 * delta**2)
    if not unbiased:
        score /= 64
    score = 0.5 / n / bw / np.sqrt(np.pi) + score / n**2 / bw / np.sqrt(np.pi)
    return score


def test_histogram():
    school = load_arviz_data("non_centered_eight").posterior["mu"].values
    k_count_az, k_dens_az, _ = histogram(school, bins=np.asarray([-np.inf, 0.5, 0.7, 1, np.inf]))
    k_dens_np, *_ = np.histogram(school, bins=[-np.inf, 0.5, 0.7, 1, np.inf], density=True)
    k_count_np, *_ = np.histogram(school, bins=[-np.inf, 0.5, 0.7, 1, np.inf], density=False)
    assert np.allclose(k_count_az, k_count_np)
    assert np.allclose(k_dens_az, k_dens_np)


@pytest.mark.parametrize("unbiased", [True, False])
@pytest.mark.parametrize("bw", [0.1, 0.5, 2.0])
@pytest.mark.parametrize("n", [100, 1_000])
def test_compute_cv_score(bw, unbiased, n, seed=42):
    """Test that the histogram-based CV score matches the explicit CV score."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x_std = x.std()
    grid_counts, grid_edges = np.histogram(
        x, bins=100, range=(x.min() - 0.5 * x_std, x.max() + 0.5 * x_std)
    )
    bin_width = grid_edges[1] - grid_edges[0]
    grid = grid_edges[:-1] + 0.5 * bin_width

    # if data is discretized to regularly-spaced bins, then explicit CV score should match
    # the histogram-based CV score
    x_discrete = np.repeat(grid, grid_counts)
    rng.shuffle(x_discrete)
    score_inputs = _prepare_cv_score_inputs(grid_counts, n)
    score = _compute_cv_score(bw, n, bin_width, unbiased, *score_inputs)
    score_explicit = compute_cv_score_explicit(bw, x_discrete, unbiased)
    assert np.isclose(score, score_explicit)


@pytest.mark.parametrize("unbiased", [True, False])
def test_bw_cv_normal(unbiased, seed=42, bins=512, n=100_000):
    """Test that for normal target, selected CV bandwidth converges to known optimum."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x_std = x.std()
    grid_counts, grid_edges = np.histogram(
        x, bins=bins, range=(x.min() - 0.5 * x_std, x.max() + 0.5 * x_std)
    )
    bin_width = grid_edges[1] - grid_edges[0]
    bw = _bw_cv(x, unbiased=unbiased, bin_width=bin_width, grid_counts=grid_counts)
    assert bw > bin_width / (2 * np.pi)
    assert bw < _bw_oversmoothed(x)
    assert np.isclose(bw, _bw_scott(x), rtol=0.2)
