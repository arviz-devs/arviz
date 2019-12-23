# pylint: disable=redefined-outer-name
import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray


from ..data import load_arviz_data, datasets
from ..stats import compare
from ..rcparams import (
    rcParams,
    rc_context,
    _make_validate_choice,
    _validate_positive_int_or_none,
    _validate_probability,
    read_rcfile,
)

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
    with rc_context(fname=os.path.join(path, "test.rcparams")):
        assert rcParams["data.load"] == "eager"
    assert rcParams["data.load"] == "lazy"


def test_bad_rc_file():
    """Test bad value raises error."""
    path = os.path.dirname(os.path.abspath(__file__))
    with pytest.raises(ValueError, match="Bad val "):
        read_rcfile(os.path.join(path, "bad.rcparams"))


def test_warning_rc_file(caplog):
    """Test invalid lines and duplicated keys log warnings and bad value raises error."""
    path = os.path.dirname(os.path.abspath(__file__))
    read_rcfile(os.path.join(path, "test.rcparams"))
    records = caplog.records
    assert len(records) == 1
    assert records[0].levelname == "WARNING"
    assert "Duplicate key" in caplog.text


def test_bad_key():
    """Test the error when using unexistent keys in rcParams is correct."""
    with pytest.raises(KeyError, match="bad_key is not a valid rc"):
        rcParams["bad_key"] = "nothing"


def test_del_key_error():
    """Check that rcParams keys cannot be deleted."""
    with pytest.raises(TypeError, match="keys cannot be deleted"):
        del rcParams["data.load"]


def test_clear_error():
    """Check that rcParams cannot be cleared."""
    with pytest.raises(TypeError, match="keys cannot be deleted"):
        rcParams.clear()


def test_pop_error():
    """Check rcParams pop error."""
    with pytest.raises(TypeError, match=r"keys cannot be deleted.*get\(key\)"):
        rcParams.pop("data.load")


def test_popitem_error():
    """Check rcParams popitem error."""
    with pytest.raises(TypeError, match=r"keys cannot be deleted.*get\(key\)"):
        rcParams.popitem()


def test_setdefaults_error():
    """Check rcParams popitem error."""
    with pytest.raises(TypeError, match="Use arvizrc"):
        rcParams.setdefault("data.load", "eager")


def test_rcparams_find_all():
    data_rcparams = rcParams.find_all("data")
    assert len(data_rcparams)


def test_rcparams_repr_str():
    """Check both repr and str print all keys."""
    repr_str = rcParams.__repr__()
    str_str = rcParams.__str__()
    assert repr_str.startswith("RcParams")
    for string in (repr_str, str_str):
        assert all([key in string for key in rcParams.keys()])


### Test arvizrc.template file is up to date ###
def test_rctemplate_updated():
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../arvizrc.template")
    rc_pars_template = read_rcfile(fname)
    assert all([key in rc_pars_template.keys() for key in rcParams.keys()])
    assert all([value == rc_pars_template[key] for key, value in rcParams.items()])


### Test validation functions ###
@pytest.mark.parametrize("param", ["data.load", "stats.information_criterion"])
def test_choice_bad_values(param):
    """Test error messages are correct for rcParams validated with _make_validate_choice."""
    msg = "{}: bad_value is not one of".format(param.replace(".", r"\."))
    with pytest.raises(ValueError, match=msg):
        rcParams[param] = "bad_value"


@pytest.mark.parametrize("allow_none", (True, False))
@pytest.mark.parametrize("typeof", (str, int))
@pytest.mark.parametrize("args", [("not one", 10), (False, None), (False, 4)])
def test_make_validate_choice(args, allow_none, typeof):
    accepted_values = set(typeof(value) for value in (0, 1, 4, 6))
    validate_choice = _make_validate_choice(accepted_values, allow_none=allow_none, typeof=typeof)
    raise_error, value = args
    if value is None and not allow_none:
        raise_error = "not one of" if typeof == str else "Could not convert"
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            validate_choice(value)
    else:
        value = validate_choice(value)
        assert value in accepted_values or value is None


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
        value = _validate_positive_int_or_none(value)
        assert isinstance(value, int) or value is None


@pytest.mark.parametrize(
    "args",
    [
        ("Only.+between 0 and 1", -1),
        ("Only.+between 0 and 1", "1.3"),
        ("not convert to float", "word"),
        (False, "0.6"),
        (False, 0),
        (False, 1),
    ],
)
def test_validate_probability(args):
    raise_error, value = args
    if raise_error:
        with pytest.raises(ValueError, match=raise_error):
            _validate_probability(value)
    else:
        value = _validate_probability(value)
        assert isinstance(value, float)


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


def test_http_type_request(models, monkeypatch):
    def _urlretrive(url, _):
        raise Exception("URL Retrieved: {}".format(url))

    # Hijack url retrieve to inspect url passed
    monkeypatch.setattr(datasets, "urlretrieve", _urlretrive)

    # Test HTTPS default
    with pytest.raises(Exception) as error:
        datasets.load_arviz_data("radon")
        assert "https://" in str(error)

    # Test HTTP setting
    with pytest.raises(Exception) as error:
        rcParams["data.http_protocol"] = "http"
        datasets.load_arviz_data("radon")
        assert "http://" in str(error)
