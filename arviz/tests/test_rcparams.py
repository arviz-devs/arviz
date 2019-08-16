# pylint: disable=redefined-outer-name
import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray


from ..data import load_arviz_data
from ..stats import compare
from ..rcparams import rcParams, rc_context, _validate_positive_int_or_none, read_rcfile

from .helpers import models  # pylint: disable=unused-import


### Test rcparams classes ###
def test_rc_context_dict():
    rcParams["data.load"] = "lazy"
    with rc_context(rc={"data.load": "eager"}):
        assert rcParams["data.load"] == "eager"
    assert rcParams["data.load"] == "lazy"


def test_rc_context_file():
    path = os.path.dirname(os.path.abspath(__file__))
    rcParams["data.load"] = "lazy"
    with rc_context(fname=path + "/test.rcparams"):
        assert rcParams["data.load"] == "eager"
    assert rcParams["data.load"] == "lazy"


def test_bad_key():
    """Test the error when using unexistent keys in rcParams is correct."""
    with pytest.raises(KeyError, match="bad_key is not a valid rc"):
        rcParams["bad_key"] = "nothing"


def test_del_key_error():
    """Check that rcParams keys cannot be deleted."""
    with pytest.raises(TypeError, match="keys cannot be deleted"):
        del rcParams["data.load"]


def test_rcparams_find_all():
    data_rcparams = rcParams.find_all("data")
    assert len(data_rcparams)


### Test arvizrc.template file is up to date ###
def test_rctemplate_updated():
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../arvizrc.template")
    rc_pars_template = read_rcfile(fname)
    assert all(key in rc_pars_template.keys() for key in rcParams.keys())
    assert all(value == rc_pars_template[key] for key, value in rcParams.items())


### Test validation functions ###
@pytest.mark.parametrize("param", ["data.load", "stats.information_criterion"])
def test_choice_bad_values(param):
    """Test error messages are correct for rcParams validated with _make_validate_choice."""
    msg = "{}: bad_value is not one of".format(param.replace(".", r"\."))
    with pytest.raises(ValueError, match=msg):
        rcParams[param] = "bad_value"


@pytest.mark.parametrize(
    "args",
    [("Only positive", -1), ("Could not convert", "1.3"), (False, "2"), (False, None), (False, 1)],
)
def test_validate_positive_int_or_none(args):
    raise_error, value = args
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            _validate_positive_int_or_none(value)
    else:
        _validate_positive_int_or_none(value)


### Test integration of rcParams in ArviZ ###
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
