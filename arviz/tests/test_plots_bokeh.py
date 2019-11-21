"""Tests use the 'bokeh' backend."""
# pylint: disable=redefined-outer-name,too-many-lines
from pandas import DataFrame
import numpy as np
import pytest

from .helpers import (  # pylint: disable=unused-import
    eight_schools_params,
    models,
    create_model,
    multidim_models,
    create_multidimensional_model,
)
from ..rcparams import rcParams, rc_context
from ..plots import (
    plot_autocorr,
    plot_trace,
    plot_kde,
    plot_dist,
)

rcParams["data.load"] = "eager"


@pytest.fixture(scope="module")
def data(eight_schools_params):
    data = eight_schools_params
    return data


@pytest.fixture(scope="module")
def df_trace():
    return DataFrame({"a": np.random.poisson(2.3, 100)})


@pytest.fixture(scope="module")
def discrete_model():
    """Simple fixture for random discrete model"""
    return {"x": np.random.randint(10, size=100), "y": np.random.randint(10, size=100)}


@pytest.fixture(scope="module")
def continuous_model():
    """Simple fixture for random continuous model"""
    return {"x": np.random.beta(2, 5, size=100), "y": np.random.beta(2, 5, size=100)}


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_names": "mu"},
        {"var_names": ["mu", "tau"]},
        {"combined": True},
        {"compact": True},
        {"combined": True, "compact": True, "legend": True},
        {"divergences": "top"},
        {"divergences": False},
        {"lines": [("mu", {}, [1, 2])]},
        {"lines": [("mu", {}, 8)]},
    ],
)
def test_plot_trace(models, kwargs):
    axes = plot_trace(models.model_1, backend="bokeh", show=False, **kwargs)
    assert axes.shape


def test_plot_trace_discrete(discrete_model):
    axes = plot_trace(discrete_model, backend="bokeh", show=False)
    assert axes.shape


def test_plot_trace_max_subplots_warning(models):
    with pytest.warns(SyntaxWarning):
        with rc_context(rc={"plot.max_subplots": 1}):
            axes = plot_trace(models.model_1, backend="bokeh", show=False)
    assert axes.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plot_kwargs": {"line_dash": "solid"}},
        {"contour": True, "fill_last": False},
        {
            "contour": True,
            "contourf_kwargs": {"cmap": "plasma"},
            "contour_kwargs": {"line_width": 1},
        },
        {"contour": False},
        {"contour": False, "pcolormesh_kwargs": {"cmap": "plasma"}},
    ],
)
def test_plot_kde(continuous_model, kwargs):
    axes = plot_kde(
        continuous_model["x"], continuous_model["y"], backend="bokeh", show=False, **kwargs
    )
    assert axes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cumulative": True},
        {"cumulative": True, "plot_kwargs": {"line_dash": "dashed"}},
        {"rug": True},
        {"rug": True, "rug_kwargs": {"line_alpha": 0.2}, "rotated": True},
    ],
)
def test_plot_kde_cumulative(continuous_model, kwargs):
    axes = plot_kde(continuous_model["x"], backend="bokeh", show=False, **kwargs)
    assert axes


@pytest.mark.parametrize("kwargs", [{"kind": "hist"}, {"kind": "kde"}])
def test_plot_dist(continuous_model, kwargs):
    axes = plot_dist(continuous_model["x"], backend="bokeh", show=False, **kwargs)
    assert axes


def test_plot_kde_1d(continuous_model):
    axes = plot_kde(continuous_model["y"], backend="bokeh", show=False)
    assert axes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contour": True, "fill_last": False},
        {"contour": True, "contourf_kwargs": {"cmap": "plasma"},},
        {"contour": False},
        {"contour": False, "pcolormesh_kwargs": {"cmap": "plasma"}},
    ],
)
def test_plot_kde_2d(continuous_model, kwargs):
    axes = plot_kde(continuous_model["x"], continuous_model["y"], **kwargs)
    assert axes


@pytest.mark.parametrize(
    "kwargs", [{"plot_kwargs": {"line_dash": "solid"}}, {"cumulative": True}, {"rug": True}]
)
def test_plot_kde_quantiles(continuous_model, kwargs):
    axes = plot_kde(
        continuous_model["x"], quantiles=[0.05, 0.5, 0.95], show=False, backend="bokeh", **kwargs
    )
    assert axes


def test_plot_autocorr_short_chain():
    """Check that logic for small chain defaulting doesn't cause exception"""
    chain = np.arange(10)
    axes = plot_autocorr(chain, backend="bokeh", show=False)
    assert axes


def test_plot_autocorr_uncombined(models):
    axes = plot_autocorr(models.model_1, combined=False, backend="bokeh", show=False)
    assert axes.shape[0] == 1
    max_subplots = (
        np.inf if rcParams["plot.max_subplots"] is None else rcParams["plot.max_subplots"]
    )
    assert axes.shape[1] == min(72, max_subplots)


def test_plot_autocorr_combined(models):
    axes = plot_autocorr(models.model_1, combined=True, backend="bokeh", show=False)
    assert axes.shape[0] == 1
    assert axes.shape[1] == 18


@pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
def test_plot_autocorr_var_names(models, var_names):
    axes = plot_autocorr(models.model_1, var_names=var_names, combined=True, backend="bokeh", show=False)
    assert axes.shape
