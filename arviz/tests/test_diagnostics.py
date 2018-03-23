import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from ..stats import gelman_rubin, effective_n, geweke

good_rhat = 1.1


def fake_trace(n_samples):
    """
    Creates a fake trace with 3 variables and 2 chains
    """
    a = np.repeat((1, 5, 10), n_samples)
    b = np.repeat((1, 5, 1), n_samples)
    data = np.random.beta(a, b).reshape(-1, n_samples//2)
    trace = pd.DataFrame(data.T, columns=['a', 'a', 'b', 'b', 'c', 'c'])
    return trace


def test_gelman_rubin():
    """Confirm Gelman-Rubin statistic is close to 1 for a large number of samples.
    Also checks the correct shape"""
    trace = fake_trace(1000)
    rhat = gelman_rubin(trace)
    assert all(1 / good_rhat < r < good_rhat for r in rhat.values)
    assert rhat.shape == (3,)


def test_gelman_rubin_bad():
    """Confirm Gelman-Rubin statistic is far from 1 for a small number of samples."""
    trace = fake_trace(6)
    rhat = gelman_rubin(trace)
    assert not all(1 / good_rhat < r < good_rhat for r in rhat.values)


def test_effective_n():
    n_samples = 1000
    trace = fake_trace(n_samples)
    eff_n = effective_n(trace)
    assert_allclose(eff_n, n_samples, 2)
    assert eff_n.shape == (3,)


def test_geweke():
    trace = fake_trace(1000)
    gw = geweke(trace)
    assert max(abs(gw['a'][:, 1])) < 1
    assert max(abs(gw['a'][:, 0])) > -1
