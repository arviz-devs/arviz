# pylint: disable=redefined-outer-name
import os
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.stats import gaussian_kde
import numpy as np
import pytest

from ..data import from_dict, load_arviz_data
from ..stats import compare, psislw, loo, waic
from .helpers import eight_schools_params  # pylint: disable=unused-import
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
    _fast_kde,
    plot_khat,
    plot_hpd,
    plot_dist,
    plot_rank,
    plot_elpd,
)

np.random.seed(0)


def create_model(seed=10):
    """Create model with fake data."""
    np.random.seed(seed)
    nchains = 4
    ndraws = 500
    data = {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }
    posterior = {
        "mu": np.random.randn(nchains, ndraws),
        "tau": abs(np.random.randn(nchains, ndraws)),
        "eta": np.random.randn(nchains, ndraws, data["J"]),
        "theta": np.random.randn(nchains, ndraws, data["J"]),
    }
    posterior_predictive = {"y": np.random.randn(nchains, ndraws, len(data["y"]))}
    sample_stats = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": np.random.randn(nchains, ndraws) > 0.90,
        "log_likelihood": np.random.randn(nchains, ndraws, data["J"]),
    }
    prior = {
        "mu": np.random.randn(nchains, ndraws) / 2,
        "tau": abs(np.random.randn(nchains, ndraws)) / 2,
        "eta": np.random.randn(nchains, ndraws, data["J"]) / 2,
        "theta": np.random.randn(nchains, ndraws, data["J"]) / 2,
    }
    prior_predictive = {"y": np.random.randn(nchains, ndraws, len(data["y"])) / 2}
    sample_stats_prior = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": (np.random.randn(nchains, ndraws) > 0.95).astype(int),
    }
    model = from_dict(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        sample_stats=sample_stats,
        prior=prior,
        prior_predictive=prior_predictive,
        sample_stats_prior=sample_stats_prior,
        observed_data={"y": data["y"]},
        dims={"y": ["obs_dim"], "log_likelihood": ["obs_dim"]},
        coords={"obs_dim": range(data["J"])},
    )
    return model


def create_multidimensional_model(seed=10):
    """Create model with fake data."""
    np.random.seed(seed)
    nchains = 4
    ndraws = 500
    ndim1 = 5
    ndim2 = 7
    data = {
        "y": np.random.normal(size=(ndim1, ndim2)),
        "sigma": np.random.normal(size=(ndim1, ndim2)),
    }
    posterior = {
        "mu": np.random.randn(nchains, ndraws),
        "tau": abs(np.random.randn(nchains, ndraws)),
        "eta": np.random.randn(nchains, ndraws, ndim1, ndim2),
        "theta": np.random.randn(nchains, ndraws, ndim1, ndim2),
    }
    posterior_predictive = {"y": np.random.randn(nchains, ndraws, ndim1, ndim2)}
    sample_stats = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": np.random.randn(nchains, ndraws) > 0.90,
        "log_likelihood": np.random.randn(nchains, ndraws, ndim1, ndim2),
    }
    prior = {
        "mu": np.random.randn(nchains, ndraws) / 2,
        "tau": abs(np.random.randn(nchains, ndraws)) / 2,
        "eta": np.random.randn(nchains, ndraws, ndim1, ndim2) / 2,
        "theta": np.random.randn(nchains, ndraws, ndim1, ndim2) / 2,
    }
    prior_predictive = {"y": np.random.randn(nchains, ndraws, ndim1, ndim2) / 2}
    sample_stats_prior = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": (np.random.randn(nchains, ndraws) > 0.95).astype(int),
    }
    model = from_dict(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        sample_stats=sample_stats,
        prior=prior,
        prior_predictive=prior_predictive,
        sample_stats_prior=sample_stats_prior,
        observed_data={"y": data["y"]},
        dims={"y": ["dim1", "dim2"], "log_likelihood": ["dim1", "dim2"]},
        coords={"dim1": range(ndim1), "dim2": range(ndim2)},
    )
    return model


@pytest.fixture(scope="module")
def models():
    class Models:
        model_1 = create_model(seed=10)
        model_2 = create_model(seed=11)

    return Models()


@pytest.fixture(scope="module")
def multidim_models():
    class Models:
        model_1 = create_multidimensional_model(seed=10)
        model_2 = create_multidimensional_model(seed=11)

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
    obj = [getattr(models, model_fit) for model_fit in ["model_1", "model_2"]]
    axes = plot_density(obj, **kwargs)
    assert axes.shape[0] >= 18


def test_plot_density_discrete(discrete_model):
    axes = plot_density(discrete_model, shade=0.9)
    assert axes.shape[0] == 2


def test_plot_density_bad_kwargs(models):
    obj = [getattr(models, model_fit) for model_fit in ["model_1", "model_2"]]
    with pytest.raises(ValueError):
        plot_density(obj, point_estimate="bad_value")

    with pytest.raises(ValueError):
        plot_density(obj, data_labels=["bad_value_{}".format(i) for i in range(len(obj) + 10)])

    with pytest.raises(ValueError):
        plot_density(obj, credible_interval=2)


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
    axes = plot_trace(models.model_1, **kwargs)
    assert axes.shape


def test_plot_trace_discrete(discrete_model):
    axes = plot_trace(discrete_model)
    assert axes.shape


def test_plot_trace_max_plots_warning(models):
    with pytest.warns(SyntaxWarning):
        axes = plot_trace(models.model_1, max_plots=1)
    assert axes.shape


@pytest.mark.parametrize("model_fits", [["model_1"], ["model_1", "model_2"]])
@pytest.mark.parametrize(
    "args_expected",
    [
        ({}, 1),
        ({"var_names": "mu"}, 1),
        ({"var_names": "mu", "rope": (-1, 1)}, 1),
        ({"r_hat": True, "quartiles": False}, 2),
        ({"var_names": ["mu"], "colors": "C0", "ess": True, "combined": True}, 2),
        ({"kind": "ridgeplot", "r_hat": True, "ess": True}, 3),
        ({"kind": "ridgeplot", "r_hat": True, "ess": True, "ridgeplot_alpha": 0}, 3),
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


@pytest.mark.parametrize("model_fits", [["model_1"], ["model_1", "model_2"]])
def test_plot_forest_bad(models, model_fits):
    obj = [getattr(models, model_fit) for model_fit in model_fits]
    with pytest.raises(TypeError):
        plot_forest(obj, kind="bad_kind")

    with pytest.raises(ValueError):
        plot_forest(obj, model_names=["model_name_{}".format(i) for i in range(len(obj) + 10)])


@pytest.mark.parametrize("kind", ["kde", "hist"])
def test_plot_energy(models, kind):
    assert plot_energy(models.model_1, kind=kind)


def test_plot_energy_bad(models):
    with pytest.raises(ValueError):
        plot_energy(models.model_1, kind="bad_kind")


def test_plot_parallel_raises_valueerror(df_trace):  # pylint: disable=invalid-name
    with pytest.raises(ValueError):
        plot_parallel(df_trace)


@pytest.mark.parametrize("norm_method", [None, "normal", "minmax", "rank"])
def test_plot_parallel(models, norm_method):
    assert plot_parallel(models.model_1, var_names=["mu", "tau"], norm_method=norm_method)


@pytest.mark.parametrize("var_names", [None, "mu", ["mu", "tau"]])
def test_plot_parallel_exception(models, var_names):
    """Ensure that correct exception is raised when one variable is passed."""
    with pytest.raises(ValueError):
        assert plot_parallel(models.model_1, var_names=var_names, norm_method="foo")


@pytest.mark.parametrize("kind", ["scatter", "hexbin", "kde"])
def test_plot_joint(models, kind):
    axjoin, _, _ = plot_joint(models.model_1, var_names=("mu", "tau"), kind=kind)
    assert axjoin


def test_plot_joint_discrete(discrete_model):
    axjoin, _, _ = plot_joint(discrete_model)
    assert axjoin


def test_plot_joint_bad(models):
    with pytest.raises(ValueError):
        plot_joint(models.model_1, var_names=("mu", "tau"), kind="bad_kind")

    with pytest.raises(Exception):
        plot_joint(models.model_1, var_names=("mu", "tau", "eta"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plot_kwargs": {"linestyle": "-"}},
        {"contour": True, "fill_last": False},
        {
            "contour": True,
            "contourf_kwargs": {"cmap": "plasma"},
            "contour_kwargs": {"linewidths": 1},
        },
        {"contour": False},
        {"contour": False, "pcolormesh_kwargs": {"cmap": "plasma"}},
    ],
)
def test_plot_kde(continuous_model, kwargs):
    axes = plot_kde(continuous_model["x"], continuous_model["y"], **kwargs)
    assert axes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cumulative": True},
        {"cumulative": True, "plot_kwargs": {"linestyle": "--"}},
        {"rug": True},
        {"rug": True, "rug_kwargs": {"alpha": 0.2}},
    ],
)
def test_plot_kde_cumulative(continuous_model, kwargs):
    axes = plot_kde(continuous_model["x"], quantiles=[0.25, 0.5, 0.75], **kwargs)
    assert axes


@pytest.mark.parametrize("kwargs", [{"kind": "hist"}, {"kind": "dist"}])
def test_plot_dist(continuous_model, kwargs):
    axes = plot_dist(continuous_model["x"], **kwargs)
    assert axes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plot_kwargs": {"linestyle": "-"}},
        {"contour": True, "fill_last": False},
        {"contour": False},
    ],
)
def test_plot_dist_2d_kde(continuous_model, kwargs):
    axes = plot_dist(continuous_model["x"], continuous_model["y"], **kwargs)
    assert axes


@pytest.mark.parametrize(
    "kwargs", [{"plot_kwargs": {"linestyle": "-"}}, {"cumulative": True}, {"rug": True}]
)
def test_plot_kde_quantiles(continuous_model, kwargs):
    axes = plot_kde(continuous_model["x"], **kwargs)
    assert axes


def test_plot_kde_inference_data(models):
    """
    Ensure that an exception is raised when plot_kde
    is used with an inference data or Xarray dataset object.
    """
    with pytest.raises(ValueError, match="Inference Data"):
        plot_kde(models.model_1)
    with pytest.raises(ValueError, match="Xarray"):
        plot_kde(models.model_1.posterior)


def test_plot_khat():
    linewidth = np.random.randn(20000, 10)
    _, khats = psislw(linewidth)
    axes = plot_khat(khats)
    assert axes


@pytest.mark.slow
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
def test_plot_pair(models, kwargs):
    ax = plot_pair(models.model_1, **kwargs)
    assert np.all(ax)


@pytest.mark.parametrize(
    "kwargs", [{"kind": "scatter"}, {"kind": "kde"}, {"kind": "hexbin", "colorbar": True}]
)
def test_plot_pair_2var(discrete_model, fig_ax, kwargs):
    _, ax = fig_ax
    ax = plot_pair(discrete_model, ax=ax, **kwargs)
    assert ax


def test_plot_pair_bad(models):
    with pytest.raises(ValueError):
        plot_pair(models.model_1, kind="bad_kind")
    with pytest.raises(Exception):
        plot_pair(models.model_1, var_names=["mu"])


@pytest.mark.parametrize("has_sample_stats", [True, False])
def test_plot_pair_divergences_warning(has_sample_stats):
    data = load_arviz_data("centered_eight")
    if has_sample_stats:
        # sample_stats present, diverging field missing
        data.sample_stats = data.sample_stats.rename({"diverging": "diverging_missing"})
    else:
        # sample_stats missing
        data = data.posterior  # pylint: disable=no-member
    with pytest.warns(SyntaxWarning):
        ax = plot_pair(data, divergences=True)
    assert np.all(ax)


@pytest.mark.parametrize("kind", ["density", "cumulative", "scatter"])
@pytest.mark.parametrize("alpha", [None, 0.2, 1])
@pytest.mark.parametrize("animated", [False, True])
def test_plot_ppc(models, kind, alpha, animated):
    animation_kwargs = {"blit": False}
    axes = plot_ppc(
        models.model_1,
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
def test_plot_ppc_save_animation(models, kind):
    animation_kwargs = {"blit": False}
    axes, anim = plot_ppc(
        models.model_1,
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
def test_non_linux_blit(models, monkeypatch, system, caplog):
    import platform

    def mock_system():
        return system

    monkeypatch.setattr(platform, "system", mock_system)

    animation_kwargs = {"blit": True}
    axes, anim = plot_ppc(
        models.model_1,
        kind="density",
        animated=True,
        animation_kwargs=animation_kwargs,
        num_pp_samples=5,
        random_seed=3,
    )
    records = caplog.records
    assert len(records) == 1
    assert records[0].levelname == "WARNING"
    assert axes
    assert anim


def test_plot_ppc_grid(models):
    axes = plot_ppc(models.model_1, kind="scatter", flatten=[])
    assert len(axes) == 8
    axes = plot_ppc(models.model_1, kind="scatter", flatten=[], coords={"obs_dim": [1, 2, 3]})
    assert len(axes) == 3
    axes = plot_ppc(
        models.model_1, kind="scatter", flatten=["obs_dim"], coords={"obs_dim": [1, 2, 3]}
    )
    assert len(axes) == 1


@pytest.mark.parametrize("kind", ["density", "cumulative", "scatter"])
def test_plot_ppc_bad(models, kind):
    data = from_dict(posterior={"mu": np.random.randn()})
    with pytest.raises(TypeError):
        plot_ppc(data, kind=kind)
    with pytest.raises(TypeError):
        plot_ppc(models.model_1, kind="bad_val")
    with pytest.raises(TypeError):
        plot_ppc(models.model_1, num_pp_samples="bad_val")


@pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
def test_plot_violin(models, var_names):
    axes = plot_violin(models.model_1, var_names=var_names)
    assert axes.shape


def test_plot_violin_ax(models):
    _, ax = plt.subplots(1)
    axes = plot_violin(models.model_1, var_names="mu", ax=ax)
    assert axes.shape


def test_plot_violin_layout(models):
    axes = plot_violin(models.model_1, var_names=["mu", "tau"], sharey=False)
    assert axes.shape


def test_plot_violin_discrete(discrete_model):
    axes = plot_violin(discrete_model)
    assert axes.shape


def test_plot_autocorr_short_chain():
    """Check that logic for small chain defaulting doesn't cause exception"""
    chain = np.arange(10)
    axes = plot_autocorr(chain)
    assert axes


def test_plot_autocorr_uncombined(models):
    axes = plot_autocorr(models.model_1, combined=False)
    assert axes.shape[0] == 1
    assert axes.shape[1] == 72


def test_plot_autocorr_combined(models):
    axes = plot_autocorr(models.model_1, combined=True)
    assert axes.shape[0] == 1
    assert axes.shape[1] == 18


@pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
def test_plot_autocorr_var_names(models, var_names):
    axes = plot_autocorr(models.model_1, var_names=var_names, combined=True)
    assert axes.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_names": "mu"},
        {"var_names": ("mu", "tau"), "coords": {"theta_dim_0": [0, 1]}},
        {"var_names": "mu", "ref_line": True},
        {"var_names": "mu", "ref_line": False},
    ],
)
def test_plot_rank(models, kwargs):
    axes = plot_rank(models.model_1, **kwargs)
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
def test_plot_posterior(models, kwargs):
    axes = plot_posterior(models.model_1, **kwargs)
    assert axes.shape


@pytest.mark.parametrize("kwargs", [{}, {"point_estimate": "mode"}, {"bins": None, "kind": "hist"}])
def test_plot_posterior_discrete(discrete_model, kwargs):
    axes = plot_posterior(discrete_model, **kwargs)
    assert axes.shape


def test_plot_posterior_bad(models):
    with pytest.raises(ValueError):
        plot_posterior(models.model_1, rope="bad_value")
    with pytest.raises(ValueError):
        plot_posterior(models.model_1, ref_val="bad_value")
    with pytest.raises(ValueError):
        plot_posterior(models.model_1, point_estimate="bad_value")


@pytest.mark.parametrize("point_estimate", ("mode", "mean", "median"))
def test_point_estimates(models, point_estimate):
    axes = plot_posterior(models.model_1, var_names=("mu", "tau"), point_estimate=point_estimate)
    assert axes.shape == (2,)


@pytest.mark.parametrize(
    "kwargs", [{"insample_dev": False}, {"plot_standard_error": False}, {"plot_ic_diff": False}]
)
def test_plot_compare(models, kwargs):

    model_compare = compare({"Model 1": models.model_1, "Model 2": models.model_2})

    axes = plot_compare(model_compare, **kwargs)
    assert axes


def test_plot_compare_manual(models):
    """Test compare plot without scale column"""
    model_compare = compare({"Model 1": models.model_1, "Model 2": models.model_2})

    # remove "scale" column
    del model_compare["waic_scale"]
    axes = plot_compare(model_compare)
    assert axes


def test_plot_compare_no_ic(models):
    """Check exception is raised if model_compare doesn't contain a valid information criterion"""
    model_compare = compare({"Model 1": models.model_1, "Model 2": models.model_2})

    # Drop column needed for plotting
    model_compare = model_compare.drop("waic", axis=1)
    with pytest.raises(ValueError) as err:
        plot_compare(model_compare)

    assert "comp_df must contain one of the following" in str(err)
    assert "['waic', 'loo']" in str(err)


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
def test_plot_hpd(models, data, kwargs):
    plot_hpd(data["y"], models.model_1.posterior["theta"], **kwargs)


@pytest.mark.parametrize("limits", [(-10.0, 10.0), (-5, 5), (None, None)])
def test_fast_kde_scipy(limits):
    data = np.random.normal(0, 1, 1000)
    if limits[0] is None:
        x = np.linspace(data.min(), data.max(), 200)  # pylint: disable=no-member
    else:
        x = np.linspace(*limits, 500)
    density = gaussian_kde(data).evaluate(x)
    density_fast = _fast_kde(data, xmin=limits[0], xmax=limits[1])[0]

    np.testing.assert_almost_equal(density_fast.sum(), density.sum(), 1)


@pytest.mark.parametrize("limits", [(-10.0, 10.0), (-5, 5), (None, None)])
def test_fast_kde_cumulative(limits):
    data = np.random.normal(0, 1, 1000)
    density_fast = _fast_kde(data, xmin=limits[0], xmax=limits[1], cumulative=True)[0]
    np.testing.assert_almost_equal(round(density_fast[-1], 3), 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"ic": "loo"},
        {"xlabels": True, "scale": "log"},
        {"color": "obs_dim", "xlabels": True},
        {"color": "obs_dim", "legend": True},
        {"ic": "loo", "color": "blue", "coords": {"obs_dim": slice(2, 5)}},
        {"color": np.random.uniform(size=8), "threshold": 0.1},
    ],
)
@pytest.mark.parametrize("add_model", [False, True])
@pytest.mark.parametrize("use_elpddata", [False, True])
def test_plot_elpd(models, add_model, use_elpddata, kwargs):
    model_dict = {"Model 1": models.model_1, "Model 2": models.model_2}
    if add_model:
        model_dict["Model 3"] = create_model(seed=12)

    if use_elpddata:
        ic = kwargs.get("ic", "waic")
        scale = kwargs.get("scale", "deviance")
        if ic == "waic":
            model_dict = {k: waic(v, scale=scale, pointwise=True) for k, v in model_dict.items()}
        else:
            model_dict = {k: loo(v, scale=scale, pointwise=True) for k, v in model_dict.items()}

    axes = plot_elpd(model_dict, **kwargs)
    assert np.all(axes)
    if add_model:
        assert axes.shape[0] == axes.shape[1]
        assert axes.shape[0] == len(model_dict) - 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"ic": "loo"},
        {"xlabels": True, "scale": "log"},
        {"color": "dim1", "xlabels": True},
        {"color": "dim2", "legend": True},
        {"ic": "loo", "color": "blue", "coords": {"dim2": slice(2, 4)}},
        {"color": np.random.uniform(size=35), "threshold": 0.1},
    ],
)
@pytest.mark.parametrize("add_model", [False, True])
@pytest.mark.parametrize("use_elpddata", [False, True])
def test_plot_elpd_multidim(multidim_models, add_model, use_elpddata, kwargs):
    model_dict = {"Model 1": multidim_models.model_1, "Model 2": multidim_models.model_2}
    if add_model:
        model_dict["Model 3"] = create_multidimensional_model(seed=12)

    if use_elpddata:
        ic = kwargs.get("ic", "waic")
        scale = kwargs.get("scale", "deviance")
        if ic == "waic":
            model_dict = {k: waic(v, scale=scale, pointwise=True) for k, v in model_dict.items()}
        else:
            model_dict = {k: loo(v, scale=scale, pointwise=True) for k, v in model_dict.items()}

    axes = plot_elpd(model_dict, **kwargs)
    assert np.all(axes)
    if add_model:
        assert axes.shape[0] == axes.shape[1]
        assert axes.shape[0] == len(model_dict) - 1


def test_plot_elpd_bad_ic(models):
    model_dict = {
        "Model 1": waic(models.model_1, pointwise=True),
        "Model 2": loo(models.model_2, pointwise=True),
    }
    with pytest.raises(ValueError):
        plot_elpd(model_dict, ic="bad_ic")


def test_plot_elpd_ic_error(models):
    model_dict = {
        "Model 1": waic(models.model_1, pointwise=True),
        "Model 2": loo(models.model_2, pointwise=True),
    }
    with pytest.raises(SyntaxError):
        plot_elpd(model_dict)


def test_plot_elpd_scale_error(models):
    model_dict = {
        "Model 1": waic(models.model_1, pointwise=True),
        "Model 2": waic(models.model_2, pointwise=True, scale="log"),
    }
    with pytest.raises(SyntaxError):
        plot_elpd(model_dict)


def test_plot_elpd_one_model(models):
    model_dict = {"Model 1": models.model_1}
    with pytest.raises(Exception):
        plot_elpd(model_dict)
