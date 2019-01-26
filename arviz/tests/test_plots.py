# pylint: disable=redefined-outer-name
import os
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
import pytest
import pymc3 as pm


from ..data import from_dict, from_pymc3
from ..stats import compare, psislw
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

np.random.seed(0)


@pytest.fixture(scope="module")
def models(eight_schools_params):
    class Models:
        models = load_cached_models(eight_schools_params, draws=500, chains=2)
        pymc3_model, pymc3_fit = models["pymc3"]
        stan_model, stan_fit = models["pystan"]
        emcee_fit = models["emcee"]
        pyro_fit = models["pyro"]

    return Models()


@pytest.fixture(scope="function", autouse=True)
def clean_plots(request, save_figs):
    """Close plots after each test, optionally save if --save is specified during test invocation"""

    def fin():
        if save_figs is not None:
            plt.savefig("{0}.png".format(os.path.join(save_figs, request.node.name)))
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


@pytest.fixture(scope="module")
def continuous_model():
    """Simple fixture for random continuous model"""
    return {"x": np.random.beta(2, 5, size=100), "y": np.random.beta(2, 5, size=100)}


@pytest.fixture(scope="function")
def fig_ax():
    fig, ax = plt.subplots(1, 1)
    return fig, ax


@pytest.mark.parametrize(
    "kwargs",
    [
        {"point_estimate": "mean"},
        {"point_estimate": "median"},
        {"credible_interval": 0.94},
        {"credible_interval": 1},
        {"outline": True},
        {"colors": ["g", "b", "r", "y"]},
        {"colors": "k"},
        {"hpd_markers": ["v"]},
        {"shade": 1},
    ],
)
def test_plot_density_float(models, kwargs):
    obj = [getattr(models, model_fit) for model_fit in ["pymc3_fit", "stan_fit", "pyro_fit"]]
    axes = plot_density(obj, **kwargs)
    assert axes.shape[0] >= 18


def test_plot_density_discrete(discrete_model):
    axes = plot_density(discrete_model, shade=0.9)
    assert axes.shape[0] == 2


def test_plot_density_bad_kwargs(models):
    obj = [getattr(models, model_fit) for model_fit in ["pymc3_fit", "stan_fit", "pyro_fit"]]
    with pytest.raises(ValueError):
        plot_density(obj, point_estimate="bad_value")

    with pytest.raises(ValueError):
        plot_density(obj, data_labels=["bad_value_{}".format(i) for i in range(len(obj) + 10)])

    with pytest.raises(ValueError):
        plot_density(obj, credible_interval=2)


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit"])
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
        {"lines": [("mu", {}, 8)]},
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
    "model_fits", [["pyro_fit"], ["pymc3_fit"], ["stan_fit"], ["pymc3_fit", "stan_fit"]]
)
@pytest.mark.parametrize(
    "args_expected",
    [
        ({}, 1),
        ({"var_names": "mu"}, 1),
        ({"var_names": "mu", "rope": (-1, 1)}, 1),
        ({"r_hat": True, "quartiles": False}, 2),
        ({"var_names": ["mu"], "colors": "C0", "eff_n": True, "combined": True}, 2),
        ({"kind": "ridgeplot", "r_hat": True, "eff_n": True}, 3),
        ({"kind": "ridgeplot", "r_hat": True, "eff_n": True, "ridgeplot_alpha": 0}, 3),
        (
            {
                "var_names": ["mu", "tau"],
                "rope": {"mu": [{"rope": (-0.1, 0.1)}], "tau": [{"rope": (0.2, 0.5)}]},
            },
            1,
        ),
    ],
)
def test_plot_forest(models, model_fits, args_expected):
    obj = [getattr(models, model_fit) for model_fit in model_fits]
    args, expected = args_expected
    _, axes = plot_forest(obj, **args)
    assert axes.shape == (expected,)


def test_plot_forest_rope_exception():
    with pytest.raises(ValueError) as err:
        plot_forest({"x": [1]}, rope="not_correct_format")
    assert "Argument `rope` must be None, a dictionary like" in str(err)


def test_plot_forest_single_value():
    _, axes = plot_forest({"x": [1]})
    assert axes.shape


@pytest.mark.parametrize(
    "model_fits", [["pyro_fit"], ["pymc3_fit"], ["stan_fit"], ["pymc3_fit", "stan_fit"]]
)
def test_plot_forest_bad(models, model_fits):
    obj = [getattr(models, model_fit) for model_fit in model_fits]
    with pytest.raises(TypeError):
        plot_forest(obj, kind="bad_kind")

    with pytest.raises(ValueError):
        plot_forest(obj, model_names=["model_name_{}".format(i) for i in range(len(obj) + 10)])


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
@pytest.mark.parametrize("kind", ["kde", "hist"])
def test_plot_energy(models, model_fit, kind):
    obj = getattr(models, model_fit)
    assert plot_energy(obj, kind=kind)


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
def test_plot_energy_bad(models, model_fit):
    obj = getattr(models, model_fit)
    with pytest.raises(ValueError):
        plot_energy(obj, kind="bad_kind")


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


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit"])
@pytest.mark.parametrize("kind", ["scatter", "hexbin", "kde"])
def test_plot_joint(models, model_fit, kind):
    obj = getattr(models, model_fit)
    axjoin, _, _ = plot_joint(obj, var_names=("mu", "tau"), kind=kind)
    assert axjoin


def test_plot_joint_discrete(discrete_model):
    axjoin, _, _ = plot_joint(discrete_model)
    assert axjoin


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
def test_plot_joint_bad(models, model_fit):
    obj = getattr(models, model_fit)
    with pytest.raises(ValueError):
        plot_joint(obj, var_names=("mu", "tau"), kind="bad_kind")

    with pytest.raises(Exception):
        plot_joint(obj, var_names=("mu", "tau", "eta"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plot_kwargs": {"linestyle": "-"}},
        {"contour": True, "fill_last": False},
        {"contour": False},
    ],
)
def test_plot_kde(continuous_model, kwargs):
    axes = plot_kde(continuous_model["x"], continuous_model["y"], **kwargs)
    assert axes


@pytest.mark.parametrize("kwargs", [{"cumulative": True}, {"rug": True}])
def test_plot_kde_cumulative(continuous_model, kwargs):
    axes = plot_kde(continuous_model["x"], quantiles=[0.25, 0.5, 0.75], **kwargs)
    assert axes


@pytest.mark.parametrize(
    "kwargs", [{"plot_kwargs": {"linestyle": "-"}}, {"cumulative": True}, {"rug": True}]
)
def test_plot_kde_quantiles(continuous_model, kwargs):
    axes = plot_kde(continuous_model["x"], **kwargs)
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
        {"kind": "hexbin", "colorbar": False, "var_names": ["theta"]},
        {"kind": "hexbin", "colorbar": True, "var_names": ["theta"]},
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


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
def test_plot_pair_bad(models, model_fit):
    obj = getattr(models, model_fit)
    with pytest.raises(ValueError):
        plot_pair(obj, kind="bad_kind")
    with pytest.raises(Exception):
        plot_pair(obj, var_names=["mu"])


@pytest.mark.parametrize("kind", ["density", "cumulative", "scatter"])
@pytest.mark.parametrize("alpha", [None, 0.2, 1])
@pytest.mark.parametrize("animated", [False, True])
def test_plot_ppc(models, pymc3_sample_ppc, kind, alpha, animated):
    data = from_pymc3(trace=models.pymc3_fit, posterior_predictive=pymc3_sample_ppc)
    animation_kwargs = {"blit": False}
    axes = plot_ppc(
        data,
        kind=kind,
        alpha=alpha,
        animated=animated,
        animation_kwargs=animation_kwargs,
        random_seed=3,
    )
    if animated:
        assert axes[0]
        assert axes[1]
    assert axes


@pytest.mark.parametrize("kind", ["density", "cumulative", "scatter"])
@pytest.mark.parametrize("jitter", [None, 0, 0.1, 1, 3])
@pytest.mark.parametrize("animated", [False, True])
def test_plot_ppc_multichain(kind, jitter, animated):
    np.random.seed(23)
    data = from_dict(
        posterior_predictive={
            "x": np.random.randn(4, 100, 30),
            "y_hat": np.random.randn(4, 100, 3, 10),
        },
        observed_data={"x": np.random.randn(30), "y": np.random.randn(3, 10)},
    )
    animation_kwargs = {"blit": False}
    axes = plot_ppc(
        data,
        kind=kind,
        data_pairs={"y": "y_hat"},
        jitter=jitter,
        animated=animated,
        animation_kwargs=animation_kwargs,
        random_seed=3,
    )
    if animated:
        assert np.all(axes[0])
        assert np.all(axes[1])
    else:
        assert np.all(axes)


@pytest.mark.parametrize("kind", ["density", "cumulative", "scatter"])
@pytest.mark.parametrize("animated", [False, True])
def test_plot_ppc_discrete(kind, animated):
    data = from_dict(
        observed_data={"obs": np.random.randint(1, 100, 15)},
        posterior_predictive={"obs": np.random.randint(1, 300, (1, 20, 15))},
    )

    animation_kwargs = {"blit": False}
    axes = plot_ppc(data, kind=kind, animated=animated, animation_kwargs=animation_kwargs)
    if animated:
        assert np.all(axes[0])
        assert np.all(axes[1])
    assert axes


@pytest.mark.parametrize("kind", ["density", "cumulative", "scatter"])
def test_plot_ppc_save_animation(models, pymc3_sample_ppc, kind):
    data = from_pymc3(trace=models.pymc3_fit, posterior_predictive=pymc3_sample_ppc)
    animation_kwargs = {"blit": False}
    axes, anim = plot_ppc(
        data,
        kind=kind,
        animated=True,
        animation_kwargs=animation_kwargs,
        num_pp_samples=5,
        random_seed=3,
    )
    assert axes
    assert anim
    animations_folder = "saved_animations"
    os.makedirs(animations_folder, exist_ok=True)
    path = os.path.join(animations_folder, "ppc_{}_animation.mp4".format(kind))
    anim.save(path)
    assert os.path.exists(path)
    assert os.path.getsize(path)


@pytest.mark.parametrize("kind", ["density", "cumulative", "scatter"])
def test_plot_ppc_discrete_save_animation(kind):
    data = from_dict(
        observed_data={"obs": np.random.randint(1, 100, 15)},
        posterior_predictive={"obs": np.random.randint(1, 300, (1, 20, 15))},
    )
    animation_kwargs = {"blit": False}
    axes, anim = plot_ppc(
        data,
        kind=kind,
        animated=True,
        animation_kwargs=animation_kwargs,
        num_pp_samples=5,
        random_seed=3,
    )
    assert axes
    assert anim
    animations_folder = "saved_animations"
    os.makedirs(animations_folder, exist_ok=True)
    path = os.path.join(animations_folder, "ppc_discrete_{}_animation.mp4".format(kind))
    anim.save(path)
    assert os.path.exists(path)
    assert os.path.getsize(path)


@pytest.mark.parametrize("system", ["Windows", "Darwin"])
def test_non_linux_blit(models, pymc3_sample_ppc, monkeypatch, system):
    data = from_pymc3(trace=models.pymc3_fit, posterior_predictive=pymc3_sample_ppc)

    import platform

    def mock_system():
        return system

    monkeypatch.setattr(platform, "system", mock_system)

    animation_kwargs = {"blit": True}
    with pytest.warns(UserWarning):
        axes, anim = plot_ppc(
            data,
            kind="density",
            animated=True,
            animation_kwargs=animation_kwargs,
            num_pp_samples=5,
            random_seed=3,
        )
    assert axes
    assert anim


def test_plot_ppc_grid(models, pymc3_sample_ppc):
    data = from_pymc3(trace=models.pymc3_fit, posterior_predictive=pymc3_sample_ppc)
    axes = plot_ppc(data, kind="scatter", flatten=[])
    assert len(axes) == 8
    axes = plot_ppc(data, kind="scatter", flatten=[], coords={"obs_dim_0": [1, 2, 3]})
    assert len(axes) == 3
    axes = plot_ppc(data, kind="scatter", flatten=["obs_dim_0"], coords={"obs_dim_0": [1, 2, 3]})
    assert len(axes) == 1


@pytest.mark.parametrize("kind", ["density", "cumulative", "scatter"])
def test_plot_ppc_bad(models, pymc3_sample_ppc, kind):
    data = from_pymc3(trace=models.pymc3_fit)
    with pytest.raises(TypeError):
        plot_ppc(data, kind=kind)
    data = from_pymc3(trace=models.pymc3_fit, posterior_predictive=pymc3_sample_ppc)
    with pytest.raises(TypeError):
        plot_ppc(data, kind="bad_val")
    with pytest.raises(TypeError):
        plot_ppc(data, num_pp_samples="bad_val")


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit"])
@pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
def test_plot_violin(models, model_fit, var_names):
    obj = getattr(models, model_fit)
    axes = plot_violin(obj, var_names=var_names)
    assert axes.shape


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit"])
def test_plot_violin_ax(models, model_fit):
    obj = getattr(models, model_fit)
    _, ax = plt.subplots(1)
    axes = plot_violin(obj, var_names="mu", ax=ax)
    assert axes.shape


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit"])
def test_plot_violin_layout(models, model_fit):
    obj = getattr(models, model_fit)
    axes = plot_violin(obj, var_names=["mu", "tau"], sharey=False)
    assert axes.shape


def test_plot_violin_discrete(discrete_model):
    axes = plot_violin(discrete_model)
    assert axes.shape


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit"])
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
    )


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit"])
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
        {"point_estimate": False},
        {"ref_val": 0},
        {"ref_val": None},
        {"ref_val": {"mu": [{"ref_val": 1}]}},
        {"bins": None, "kind": "hist"},
        {"mu": {"ref_val": (-1, 1)}},
    ],
)
@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit"])
def test_plot_posterior(models, model_fit, kwargs):
    obj = getattr(models, model_fit)
    axes = plot_posterior(obj, **kwargs)
    assert axes.shape


@pytest.mark.parametrize("kwargs", [{}, {"point_estimate": "mode"}, {"bins": None, "kind": "hist"}])
def test_plot_posterior_discrete(discrete_model, kwargs):
    axes = plot_posterior(discrete_model, **kwargs)
    assert axes.shape


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit"])
def test_plot_posterior_bad(models, model_fit):
    obj = getattr(models, model_fit)
    with pytest.raises(ValueError):
        plot_posterior(obj, rope="bad_value")
    with pytest.raises(ValueError):
        plot_posterior(obj, ref_val="bad_value")
    with pytest.raises(ValueError):
        plot_posterior(obj, point_estimate="bad_value")


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit", "pyro_fit"])
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


def test_plot_compare_no_ic(models):
    """Check exception is raised if model_compare doesn't contain a valid information criterion"""
    model_compare = compare({"Pymc3": models.pymc3_fit, "Pymc3_Again": models.pymc3_fit})

    # Drop column needed for plotting
    model_compare = model_compare.drop("waic", axis=1)
    with pytest.raises(ValueError) as err:
        plot_compare(model_compare)

    assert "comp_df must contain one of the following" in str(err)
    assert "['waic', 'loo']" in str(err)


@pytest.mark.parametrize("model_fit", ["pymc3_fit", "stan_fit"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"color": "0.5", "circular": True},
        {"fill_kwargs": {"alpha": 0}},
        {"plot_kwargs": {"alpha": 0}},
        {"smooth_kwargs": {"window_length": 33, "polyorder": 5, "mode": "mirror"}},
        {"smooth": False},
    ],
)
def test_plot_hpd(models, model_fit, data, kwargs):
    obj = getattr(models, model_fit)
    plot_hpd(data["y"], obj["theta"], **kwargs)
