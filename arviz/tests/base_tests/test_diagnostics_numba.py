"""Test Diagnostic methods"""
import importlib

# pylint: disable=redefined-outer-name, no-member, too-many-public-methods
import numpy as np
import pytest

from ...data import load_arviz_data
from ...rcparams import rcParams
from ...stats import bfmi, mcse, rhat
from ...stats.diagnostics import _mc_error, ks_summary
from ...utils import Numba
from ..helpers import running_on_ci
from .test_diagnostics import data  # pylint: disable=unused-import

pytestmark = pytest.mark.skipif(  # pylint: disable=invalid-name
    (importlib.util.find_spec("numba") is None) and not running_on_ci(),
    reason="test requires numba which is not installed",
)

rcParams["data.load"] = "eager"


def test_numba_bfmi():
    """Numba test for bfmi."""
    state = Numba.numba_flag
    school = load_arviz_data("centered_eight")
    data_md = np.random.rand(100, 100, 10)
    Numba.disable_numba()
    non_numba = bfmi(school.posterior["mu"].values)
    non_numba_md = bfmi(data_md)
    Numba.enable_numba()
    with_numba = bfmi(school.posterior["mu"].values)
    with_numba_md = bfmi(data_md)
    assert np.allclose(non_numba_md, with_numba_md)
    assert np.allclose(with_numba, non_numba)
    assert state == Numba.numba_flag


@pytest.mark.parametrize("method", ("rank", "split", "folded", "z_scale", "identity"))
def test_numba_rhat(method):
    """Numba test for mcse."""
    state = Numba.numba_flag
    school = np.random.rand(100, 100)
    Numba.disable_numba()
    non_numba = rhat(school, method=method)
    Numba.enable_numba()
    with_numba = rhat(school, method=method)
    assert np.allclose(with_numba, non_numba)
    assert Numba.numba_flag == state


@pytest.mark.parametrize("method", ("mean", "sd", "quantile"))
def test_numba_mcse(method, prob=None):
    """Numba test for mcse."""
    state = Numba.numba_flag
    school = np.random.rand(100, 100)
    if method == "quantile":
        prob = 0.80
    Numba.disable_numba()
    non_numba = mcse(school, method=method, prob=prob)
    Numba.enable_numba()
    with_numba = mcse(school, method=method, prob=prob)
    assert np.allclose(with_numba, non_numba)
    assert Numba.numba_flag == state


def test_ks_summary_numba():
    """Numba test for ks_summary."""
    state = Numba.numba_flag
    data = np.random.randn(100, 100)
    Numba.disable_numba()
    non_numba = (ks_summary(data)["Count"]).values
    Numba.enable_numba()
    with_numba = (ks_summary(data)["Count"]).values
    assert np.allclose(non_numba, with_numba)
    assert Numba.numba_flag == state


@pytest.mark.parametrize("batches", (1, 20))
@pytest.mark.parametrize("circular", (True, False))
def test_mcse_error_numba(batches, circular):
    """Numba test for mcse_error."""
    data = np.random.randn(100, 100)
    state = Numba.numba_flag
    Numba.disable_numba()
    non_numba = _mc_error(data, batches=batches, circular=circular)
    Numba.enable_numba()
    with_numba = _mc_error(data, batches=batches, circular=circular)
    assert np.allclose(non_numba, with_numba)
    assert state == Numba.numba_flag
