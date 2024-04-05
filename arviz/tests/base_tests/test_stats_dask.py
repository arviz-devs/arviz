# pylint: disable=redefined-outer-name, no-member
import pytest

import numpy as np

from ...data import load_arviz_data
from ...stats import loo, psislw
from ..helpers import multidim_models, importorskip  # pylint: disable=unused-import

dask = importorskip("dask", reason="Dask specific tests")

@pytest.fixture()
def chunked_centered_eight():
    centered_eight = load_arviz_data("centered_eight")
    centered_eight.log_likelihood = centered_eight.log_likelihood.chunk({"school": 4})
    return centered_eight

@pytest.mark.parametrize("multidim", (True, False))
def test_psislw(chunked_centered_eight, multidim_models, multidim):
    if multidim:
        log_like = multidim_models.model_1.log_likelihood["y"].stack(__sample__=["chain", "draw"]).chunk({"dim2": 3})
    else:
        log_like = chunked_centered_eight.log_likelihood["obs"].stack(__sample__=["chain", "draw"])
    log_weights, khat = psislw(-log_like, dask_kwargs={"dask": "parallelized", })
    assert log_weights.shape[:-1] == khat.shape


@pytest.mark.parametrize("multidim", (True, False))
def test_loo(chunked_centered_eight, multidim_models, multidim):
    """Test approximate leave one out criterion calculation"""
    if multidim:
        idata = multidim_models.model_1
        idata.log_likelihood = idata.log_likelihood.chunk({"dim2": 3})
    else:
        idata = chunked_centered_eight
        idata.log_likelihood = idata.log_likelihood
    assert loo(idata, dask_kwargs={"dask": "parallelized"}) is not None
    loo_pointwise = loo(idata, pointwise=True, dask_kwargs={"dask": "parallelized"})
    assert loo_pointwise is not None
    assert "loo_i" in loo_pointwise
    assert "pareto_k" in loo_pointwise
    assert "scale" in loo_pointwise

def test_compare_loo(centered_eight):
    loo_ram = loo(centered_eight)
    centered_eight.log_likelihood = centered_eight.log_likelihood.chunk({"school": 2})
    loo_dask = loo(centered_eight, dask_kwargs={"dask": "parallelized"})
    assert np.isclose(loo_ram["elpd"], loo_dask["eldp"])
