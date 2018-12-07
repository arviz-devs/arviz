# pylint: disable=redefined-outer-name
import os
import time
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from pandas import DataFrame
import xarray as xr
import numpy as np
import pytest
import pymc3 as pm


from ..data import from_pymc3, InferenceData
from ..stats import compare
from .helpers import eight_schools_params, load_cached_models  # pylint: disable=unused-import
from ..plots import (
    plot_density,
    plot_trace,
    plot_energy,
    plot_posterior,
    plot_autocorr,
    plot_forest,
    plot_parallel,
    plot_pair,
    plot_joint,
    plot_ppc,
    plot_violin,
    plot_compare,
    plot_kde,
    plot_khat,
    plot_hpd,
)

from ..stats import psislw

np.random.seed(0)


@pytest.fixture(scope="module")
def models(eight_schools_params):
    class Models:
        models = load_cached_models(eight_schools_params, draws=500, chains=2)
        pymc3_model, pymc3_fit = models["pymc3"]
        stan_model, stan_fit = models["pystan"]
        emcee_fit = models["emcee"]
        pyro_fit = models["pyro"]
        tfp_fit = models["tensorflow_probability"]

    return Models()


@pytest.fixture(scope="function", autouse=True)
def clean_plots(request, save_figs):
    """Close plots after each test, optionally save if --save is specified during test invocation"""

    def fin():
        if save_figs is not None:

            # Retry save three times
            for i in range(3):
                try:
                    plt.savefig("{0}.png".format(os.path.join(save_figs, request.node.name)))

                except Exception as err:  # pylint: disable=broad-except
                    if i == 2:
                        raise err
                    time.sleep(0.250)
                else:
                    break

        plt.close("all")

    request.addfinalizer(fin)


@pytest.fixture(scope="module")
def pymc3_sample_ppc(models):

    with models.pymc3_model:
        return pm.sample_ppc(models.pymc3_fit, 100)


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


@pytest.fixture(scope="function")
def fig_ax():
    fig, ax = plt.subplots(1, 1)
    return fig, ax


@pytest.mark.parametrize(
    "kwargs",
    [
        {"point_estimate": "mean"},
        {"point_estimate": "median"},
        {"outline": True},
        {"colors": ["g", "b", "r", "y"]},
        {"hpd_markers": ["v"]},
        {"shade": 1},
    ],
)
def test_plot_density_float(models, kwargs):
    obj = [
        getattr(models, model_fit) for model_fit in ["pymc3_fit", "stan_fit", "pyro_fit", "tfp_fit"]
    ]
    axes = plot_density(obj, **kwargs)
    assert axes.shape[0] >= 18


def test_plot_density_discrete(discrete_model):
    axes = plot_density(discrete_model, shade=0.9)
    assert axes.shape[0] == 2


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit", "tfp_fit"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_names": "mu"},
        {"var_names": ["mu", "tau"]},
        {"combined": True},
        {"divergences": "top"},
        {"divergences": False},
        {"lines": [("mu", {}, [1, 2])]},
        {"lines": [("mu", 0)]},
    ],
)
def test_plot_trace(models, model_fit, kwargs):
    obj = getattr(models, model_fit)
    axes = plot_trace(obj, **kwargs)
    assert axes.shape


def test_plot_trace_emcee(models):
    axes = plot_trace(models.emcee_fit)
    assert axes.shape


def test_plot_trace_discrete(discrete_model):
    axes = plot_trace(discrete_model)
    assert axes.shape


@pytest.mark.parametrize(
    "model_fits",
    [["tfp_fit"], ["pyro_fit"], ["pymc3_fit"], ["stan_fit"], ["pymc3_fit", "stan_fit"]],
)
@pytest.mark.parametrize(
    "args_expected",
    [
        ({}, 1),
        ({"var_names": "mu"}, 1),
        ({"r_hat": True, "quartiles": False}, 2),
        ({"var_names": ["mu"], "colors": "C0", "eff_n": True, "combined": True}, 2),
        ({"kind": "ridgeplot", "r_hat": True, "eff_n": True}, 3),
    ],
)
def test_plot_forest(models, model_fits, args_expected):
    obj = [getattr(models, model_fit) for model_fit in model_fits]
    args, expected = args_expected
    _, axes = plot_forest(obj, **args)
    assert axes.shape == (expected,)


def test_plot_forest_single_value():
    _, axes = plot_forest({"x": [1]})
    assert axes.shape


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
@pytest.mark.parametrize("kind", ["kde", "hist"])
def test_plot_energy(models, model_fit, kind):
    obj = getattr(models, model_fit)
    assert plot_energy(obj, kind=kind)


def test_plot_parallel_raises_valueerror(df_trace):  # pylint: disable=invalid-name
    with pytest.raises(ValueError):
        plot_parallel(df_trace)


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
def test_plot_parallel(models, model_fit):
    obj = getattr(models, model_fit)
    assert plot_parallel(obj, var_names=["mu", "tau"])


def test_plot_parallel_exception(models):
    """Ensure that correct exception is raised when one variable is passed."""
    with pytest.raises(ValueError):
        assert plot_parallel(models.pymc3_fit, var_names="mu")


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit", "tfp_fit"])
@pytest.mark.parametrize("kind", ["scatter", "hexbin", "kde"])
def test_plot_joint(models, model_fit, kind):
    obj = getattr(models, model_fit)
    axjoin, _, _ = plot_joint(obj, var_names=("mu", "tau"), kind=kind)
    assert axjoin


def test_plot_joint_discrete(discrete_model):
    axjoin, _, _ = plot_joint(discrete_model)
    assert axjoin


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plot_kwargs": {"linestyle": "-"}},
        {"contour": True, "fill_last": False},
        {"contour": False},
    ],
)
def test_plot_kde(discrete_model, kwargs):
    axes = plot_kde(discrete_model["x"], discrete_model["y"], **kwargs)
    assert axes


@pytest.mark.parametrize(
    "kwargs", [{"plot_kwargs": {"linestyle": "-"}}, {"cumulative": True}, {"rug": True}]
)
def test_plot_kde_cumulative(discrete_model, kwargs):
    axes = plot_kde(discrete_model["x"], **kwargs)
    assert axes


def test_plot_khat():
    linewidth = np.random.randn(20000, 10)
    _, khats = psislw(linewidth)
    axes = plot_khat(khats)
    assert axes


@pytest.mark.slow
@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "var_names": "theta",
            "divergences": True,
            "coords": {"theta_dim_0": [0, 1]},
            "plot_kwargs": {"marker": "x"},
            "divergences_kwargs": {"marker": "*", "c": "C"},
        },
        {
            "divergences": True,
            "plot_kwargs": {"marker": "x"},
            "divergences_kwargs": {"marker": "*", "c": "C"},
            "var_names": ["theta", "mu"],
        },
        {"kind": "kde", "var_names": ["theta"]},
        {
            "kind": "hexbin",
            "var_names": ["theta"],
            "coords": {"theta_dim_0": [0, 1]},
            "colorbar": True,
            "plot_kwargs": {"cmap": "viridis"},
            "textsize": 20,
        },
    ],
)
def test_plot_pair(models, model_fit, kwargs):
    obj = getattr(models, model_fit)
    ax = plot_pair(obj, **kwargs)
    assert ax


@pytest.mark.parametrize(
    "kwargs", [{"kind": "scatter"}, {"kind": "kde"}, {"kind": "hexbin", "colorbar": True}]
)
def test_plot_pair_2var(discrete_model, fig_ax, kwargs):
    _, ax = fig_ax
    ax = plot_pair(discrete_model, ax=ax, **kwargs)
    assert ax


@pytest.mark.parametrize("kind", ["density", "cumulative", "scatter"])
def test_plot_ppc(models, pymc3_sample_ppc, kind):
    data = from_pymc3(trace=models.pymc3_fit, posterior_predictive=pymc3_sample_ppc)
    axes = plot_ppc(data, kind=kind)
    assert axes


@pytest.mark.parametrize("kind", ["density", "cumulative", "scatter"])
def test_plot_ppc_discrete(kind):
    data = MagicMock(spec=InferenceData)
    observed_data = xr.Dataset({"obs": (["obs_dim_0"], [9, 9])}, coords={"obs_dim_0": [1, 2]})
    posterior_predictive = xr.Dataset(
        {"obs": (["draw", "chain", "obs_dim_0"], [[[1]], [[1]]])},
        coords={"obs_dim_0": [1], "chain": [1], "draw": [1, 2]},
    )
    data.observed_data = observed_data
    data.posterior_predictive = posterior_predictive

    axes = plot_ppc(data, kind=kind)
    assert axes


def test_plot_ppc_grid(models, pymc3_sample_ppc):
    data = from_pymc3(trace=models.pymc3_fit, posterior_predictive=pymc3_sample_ppc)
    axes = plot_ppc(data, kind="scatter", flatten=[])
    assert len(axes) == 8
    axes = plot_ppc(data, kind="scatter", flatten=[], coords={"obs_dim_0": [1, 2, 3]})
    assert len(axes) == 3
    axes = plot_ppc(data, kind="scatter", flatten=["obs_dim_0"], coords={"obs_dim_0": [1, 2, 3]})
    assert len(axes) == 1


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit"])
@pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
def test_plot_violin(models, model_fit, var_names):
    obj = getattr(models, model_fit)
    axes = plot_violin(obj, var_names=var_names)
    assert axes.shape


def test_plot_violin_discrete(discrete_model):
    axes = plot_violin(discrete_model)
    assert axes.shape


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit", "tfp_fit"])
def test_plot_autocorr_uncombined(models, model_fit):
    obj = getattr(models, model_fit)
    axes = plot_autocorr(obj, combined=False)
    assert axes.shape[0] == 1
    assert (
        axes.shape[1] == 36
        and model_fit == "pymc3_fit"
        or axes.shape[1] == 68
        and model_fit == "stan_fit"
        or axes.shape[1] == 10
        and model_fit == "pyro_fit"
        or axes.shape[1] == 10
        and model_fit == "tfp_fit"
    )


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit", "tfp_fit"])
def test_plot_autocorr_combined(models, model_fit):
    obj = getattr(models, model_fit)
    axes = plot_autocorr(obj, combined=True)
    assert axes.shape[0] == 1
    assert (
        axes.shape[1] == 18
        and model_fit == "pymc3_fit"
        or axes.shape[1] == 34
        and model_fit == "stan_fit"
        or axes.shape[1] == 10
        and model_fit == "pyro_fit"
        or axes.shape[1] == 10
        and model_fit == "tfp_fit"
    )


@pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
def test_plot_autocorr_var_names(models, var_names):
    axes = plot_autocorr(models.pymc3_fit, var_names=var_names, combined=True)
    assert axes.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_names": "mu"},
        {"var_names": ("mu", "tau")},
        {"rope": (-2, 2)},
        {"rope": {"mu": [{"rope": (-2, 2)}], "theta": [{"school": "Choate", "rope": (2, 4)}]}},
        {"point_estimate": "mode"},
        {"point_estimate": "median"},
        {"ref_val": 0},
        {"bins": None, "kind": "hist"},
        {"mu": {"ref_val": (-1, 1)}},
    ],
)
@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit", "tfp_fit"])
def test_plot_posterior(models, model_fit, kwargs):
    obj = getattr(models, model_fit)
    axes = plot_posterior(obj, **kwargs)
    assert axes.shape


@pytest.mark.parametrize("kwargs", [{}, {"point_estimate": "mode"}, {"bins": None, "kind": "hist"}])
def test_plot_posterior_discrete(discrete_model, kwargs):
    axes = plot_posterior(discrete_model, **kwargs)
    assert axes.shape


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit", "tfp_fit"])
@pytest.mark.parametrize("point_estimate", ("mode", "mean", "median"))
def test_point_estimates(models, model_fit, point_estimate):
    obj = getattr(models, model_fit)
    axes = plot_posterior(obj, var_names=("mu", "tau"), point_estimate=point_estimate)
    assert axes.shape == (2,)


@pytest.mark.parametrize(
    "kwargs", [{"insample_dev": False}, {"plot_standard_error": False}, {"plot_ic_diff": False}]
)
def test_plot_compare(models, kwargs):

    # Pymc3 models create loglikelihood on InferenceData automatically
    model_compare = compare({"Pymc3": models.pymc3_fit, "Pymc3_Again": models.pymc3_fit})

    axes = plot_compare(model_compare, **kwargs)
    assert axes


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"color": "0.5", "circular": True},
        {"fill_kwargs": {"alpha": 0}},
        {"plot_kwargs": {"alpha": 0}},
        {"smooth_kwargs": {"window_length": 33, "polyorder": 5, "mode": "mirror"}},
    ],
)
def test_plot_hpd(models, model_fit, data, kwargs):
    obj = getattr(models, model_fit)
    plot_hpd(data["y"], obj["theta"], **kwargs)
