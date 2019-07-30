# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray


from ..data import load_arviz_data
from ..stats import compare
from ..rcparams import rcParams, rc_context
from .helpers import models  # pylint: disable=unused-import


def test_bad_key():
    """Test the error when using unexistent keys in rcParams is correct."""
    with pytest.raises(KeyError, match="bad_key is not a valid rc"):
        rcParams["bad_key"] = "nothing"


@pytest.mark.parametrize("param", ["data.load", "stats.information_criterion"])
def test_choice_bad_values(param):
    """Test error messages are correct for rcParams validated with _make_validate_choice."""
    msg = "{}: bad_value is not one of".format(param.replace(".", r"\."))
    with pytest.raises(ValueError, match=msg):
        rcParams[param] = "bad_value"


def test_rc_context():
    rcParams["data.load"] = "lazy"
    with rc_context(rc={"data.load": "eager"}):
        assert rcParams["data.load"] == "eager"
    assert rcParams["data.load"] == "lazy"


def test_data_load():
    rcParams["data.load"] = "lazy"
    idata_lazy = load_arviz_data("centered_eight")
    assert isinstance(
        idata_lazy.posterior.mu.variable._data,  # pylint: disable=protected-access
        MemoryCachedArray,
    )
    assert rcParams["data.load"] == "lazy"
    rcParams["data.load"] = "eager"
    idata_eager = load_arviz_data("centered_eight")
    assert isinstance(
        idata_eager.posterior.mu.variable._data, np.ndarray  # pylint: disable=protected-access
    )
    assert rcParams["data.load"] == "eager"


def test_stats_information_criterion(models):
    rcParams["stats.information_criterion"] = "waic"
    df_comp = compare({"model1": models.model_1, "model2": models.model_2})
    assert "waic" in df_comp.columns
    rcParams["stats.information_criterion"] = "loo"
    df_comp = compare({"model1": models.model_1, "model2": models.model_2})
    assert "loo" in df_comp.columns
