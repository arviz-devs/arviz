# pylint: disable=redefined-outer-name, no-member
import numpy as np
import pytest

from ...rcparams import rcParams
from ...stats import r2_score, summary
from ...utils import Numba
from ..helpers import (  # pylint: disable=unused-import
    check_multiple_attrs,
    importorskip,
    multidim_models,
)
from .test_stats import centered_eight, non_centered_eight  # pylint: disable=unused-import

numba = importorskip("numba")

rcParams["data.load"] = "eager"


@pytest.mark.parametrize("circ_var_names", [["mu"], None])
def test_summary_circ_vars(centered_eight, circ_var_names):
    assert summary(centered_eight, circ_var_names=circ_var_names) is not None
    state = Numba.numba_flag
    Numba.disable_numba()
    assert summary(centered_eight, circ_var_names=circ_var_names) is not NotImplementedError
    Numba.enable_numba()
    assert state == Numba.numba_flag


def test_numba_stats():
    """Numba test for r2_score"""
    state = Numba.numba_flag  # Store the current state of Numba
    set_1 = np.random.randn(100, 100)
    set_2 = np.random.randn(100, 100)
    set_3 = np.random.rand(100)
    set_4 = np.random.rand(100)
    Numba.disable_numba()
    non_numba = r2_score(set_1, set_2)
    non_numba_one_dimensional = r2_score(set_3, set_4)
    Numba.enable_numba()
    with_numba = r2_score(set_1, set_2)
    with_numba_one_dimensional = r2_score(set_3, set_4)
    assert state == Numba.numba_flag  # Ensure that initial state = final state
    assert np.allclose(non_numba, with_numba)
    assert np.allclose(non_numba_one_dimensional, with_numba_one_dimensional)
