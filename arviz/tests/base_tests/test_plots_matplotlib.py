"""Tests use the default backend."""

# pylint: disable=redefined-outer-name,too-many-lines
import os
import re
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib import animation
from pandas import DataFrame
from scipy.stats import gaussian_kde, norm

from ...data import from_dict, load_arviz_data
from ...labels import MapLabeller
from ...plots import (
    plot_autocorr,
    plot_bf,
    plot_bpv,
    plot_compare,
    plot_density,
    plot_dist,
    plot_dist_comparison,
    plot_dot,
    plot_ecdf,
    plot_elpd,
    plot_energy,
    plot_ess,
    plot_forest,
    plot_hdi,
    plot_kde,
    plot_khat,
    plot_lm,
    plot_loo_pit,
    plot_mcse,
    plot_pair,
    plot_parallel,
    plot_posterior,
    plot_ppc,
    plot_rank,
    plot_separation,
    plot_trace,
    plot_ts,
    plot_violin,
)
from ...plots.dotplot import wilkinson_algorithm
from ...plots.plot_utils import plot_point_interval
from ...rcparams import rc_context, rcParams
from ...stats import compare, hdi, loo, waic
from ...stats.density_utils import kde as _kde
from ...utils import BehaviourChangeWarning, _cov
from ..helpers import (  # pylint: disable=unused-import
    RandomVariableTestClass,
    create_model,
    create_multidimensional_model,
    does_not_warn,
    eight_schools_params,
    models,
    multidim_models,
)

rcParams["data.load"] = "eager"


@pytest.fixture(scope="function", autouse=True)
def clean_plots(request, save_figs):
    """Close plots after each test, optionally save if --save is specified during test invocation"""

    def fin():
        if save_figs is not None:
            plt.savefig(f"{os.path.join(save_figs, request.node.name)}.png")
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
def discrete_multidim_model():
    """Simple fixture for random discrete model"""
    idata = from_dict(
        {"x": np.random.randint(10, size=(2, 50, 3)), "y": np.random.randint(10, size=(2, 50))},
        dims={"x": ["school"]},
    )
    return idata


@pytest.fixture(scope="module")
def continuous_model():
    """Simple fixture for random continuous model"""
    return {"x": np.random.beta(2, 5, size=100), "y": np.random.beta(2, 5, size=100)}


@pytest.fixture(scope="function")
def fig_ax():
    fig, ax = plt.subplots(1, 1)
    return fig, ax


@pytest.fixture(scope="module")
def data_random():
    return np.random.randint(1, 100, size=20)


@pytest.fixture(scope="module")
def data_list():
    return list(range(11, 31))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"point_estimate": "mean"},
        {"point_estimate": "median"},
        {"hdi_prob": 0.94},
        {"hdi_prob": 1},
        {"outline": True},
        {"colors": ["g", "b", "r", "y"]},
        {"colors": "k"},
        {"hdi_markers": ["v"]},
        {"shade": 1},
        {"transform": lambda x: x + 1},
        {"ax": plt.subplots(6, 3)[1]},
    ],
)
def test_plot_density_float(models, kwargs):
    obj = [getattr(models, model_fit) for model_fit in ["model_1", "model_2"]]
    axes = plot_density(obj, **kwargs)
    assert axes.shape == (6, 3)


def test_plot_density_discrete(discrete_model):
    axes = plot_density(discrete_model, shade=0.9)
    assert axes.size == 2


def test_plot_density_no_subset():
    """Test plot_density works when variables are not subset of one another (#1093)."""
    model_ab = from_dict(
        {
            "a": np.random.normal(size=200),
            "b": np.random.normal(size=200),
        }
    )
    model_bc = from_dict(
        {
            "b": np.random.normal(size=200),
            "c": np.random.normal(size=200),
        }
    )
    axes = plot_density([model_ab, model_bc])
    assert axes.size == 3


def test_plot_density_nonstring_varnames():
    """Test plot_density works when variables are not strings."""
    rv1 = RandomVariableTestClass("a")
    rv2 = RandomVariableTestClass("b")
    rv3 = RandomVariableTestClass("c")
    model_ab = from_dict(
        {
            rv1: np.random.normal(size=200),
            rv2: np.random.normal(size=200),
        }
    )
    model_bc = from_dict(
        {
            rv2: np.random.normal(size=200),
            rv3: np.random.normal(size=200),
        }
    )
    axes = plot_density([model_ab, model_bc])
    assert axes.size == 3


def test_plot_density_bad_kwargs(models):
    obj = [getattr(models, model_fit) for model_fit in ["model_1", "model_2"]]
    with pytest.raises(ValueError):
        plot_density(obj, point_estimate="bad_value")

    with pytest.raises(ValueError):
        plot_density(obj, data_labels=[f"bad_value_{i}" for i in range(len(obj) + 10)])

    with pytest.raises(ValueError):
        plot_density(obj, hdi_prob=2)

    with pytest.raises(ValueError):
        plot_density(obj, filter_vars="bad_value")


def test_plot_density_discrete_combinedims(discrete_model):
    axes = plot_density(discrete_model, combine_dims={"school"}, shade=0.9)
    assert axes.size == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"y_hat_line": True},
        {"expected_events": True},
        {"y_hat_line_kwargs": {"linestyle": "dotted"}},
        {"exp_events_kwargs": {"marker": "o"}},
    ],
)
def test_plot_separation(kwargs):
    idata = load_arviz_data("classification10d")
    ax = plot_separation(idata=idata, y="outcome", **kwargs)
    assert ax


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_names": "mu"},
        {"var_names": ["mu", "tau"]},
        {"combined": True},
        {"compact": True},
        {"combined": True, "compact": True, "legend": True},
        {"divergences": "top", "legend": True},
        {"divergences": False},
        {"kind": "rank_vlines"},
        {"kind": "rank_bars"},
        {"lines": [("mu", {}, [1, 2])]},
        {"lines": [("mu", {}, 8)]},
        {"circ_var_names": ["mu"]},
        {"circ_var_names": ["mu"], "circ_var_units": "degrees"},
    ],
)
def test_plot_trace(models, kwargs):
    axes = plot_trace(models.model_1, **kwargs)
    assert axes.shape


@pytest.mark.parametrize(
    "compact",
    [True, False],
)
@pytest.mark.parametrize(
    "combined",
    [True, False],
)
def test_plot_trace_legend(compact, combined):
    idata = load_arviz_data("rugby")
    axes = plot_trace(
        idata, var_names=["home", "atts_star"], compact=compact, combined=combined, legend=True
    )
    assert axes[0, 1].get_legend()
    compact_legend = axes[1, 0].get_legend()
    if compact:
        assert axes.shape == (2, 2)
        assert compact_legend
    else:
        assert axes.shape == (7, 2)
        assert not compact_legend


def test_plot_trace_discrete(discrete_model):
    axes = plot_trace(discrete_model)
    assert axes.shape


def test_plot_trace_max_subplots_warning(models):
    with pytest.warns(UserWarning):
        with rc_context(rc={"plot.max_subplots": 6}):
            axes = plot_trace(models.model_1)
    assert axes.shape == (3, 2)


def test_plot_dist_comparison_warning(models):
    with pytest.warns(UserWarning):
        with rc_context(rc={"plot.max_subplots": 6}):
            axes = plot_dist_comparison(models.model_1)
    assert axes.shape == (2, 3)


@pytest.mark.parametrize("kwargs", [{"var_names": ["mu", "tau"], "lines": [("hey", {}, [1])]}])
def test_plot_trace_invalid_varname_warning(models, kwargs):
    with pytest.warns(UserWarning, match="valid var.+should be provided"):
        axes = plot_trace(models.model_1, **kwargs)
    assert axes.shape


def test_plot_trace_diverging_correctly_transposed():
    idata = load_arviz_data("centered_eight")
    idata.sample_stats["diverging"] = idata.sample_stats.diverging.T
    plot_trace(idata, divergences="bottom")


@pytest.mark.parametrize(
    "bad_kwargs", [{"var_names": ["mu", "tau"], "lines": [("mu", {}, ["hey"])]}]
)
def test_plot_trace_bad_lines_value(models, bad_kwargs):
    with pytest.raises(ValueError, match="line-positions should be numeric"):
        plot_trace(models.model_1, **bad_kwargs)


@pytest.mark.parametrize("prop", ["chain_prop", "compact_prop"])
def test_plot_trace_futurewarning(models, prop):
    with pytest.warns(FutureWarning, match=f"{prop} as a tuple.+deprecated"):
        ax = plot_trace(models.model_1, **{prop: ("ls", ("-", "--"))})
    assert ax.shape


@pytest.mark.parametrize("model_fits", [["model_1"], ["model_1", "model_2"]])
@pytest.mark.parametrize(
    "args_expected",
    [
        ({}, 1),
        ({"var_names": "mu", "transform": lambda x: x + 1}, 1),
        ({"var_names": "mu", "rope": (-1, 1), "combine_dims": {"school"}}, 1),
        ({"r_hat": True, "quartiles": False}, 2),
        ({"var_names": ["mu"], "colors": "C0", "ess": True, "combined": True}, 2),
        (
            {
                "kind": "ridgeplot",
                "ridgeplot_truncate": False,
                "ridgeplot_quantiles": [0.25, 0.5, 0.75],
            },
            1,
        ),
        ({"kind": "ridgeplot", "r_hat": True, "ess": True}, 3),
        ({"kind": "ridgeplot", "r_hat": True, "ess": True}, 3),
        ({"kind": "ridgeplot", "r_hat": True, "ess": True, "ridgeplot_alpha": 0}, 3),
        (
            {
                "var_names": ["mu", "theta"],
                "rope": {
                    "mu": [{"rope": (-0.1, 0.1)}],
                    "theta": [{"school": "Choate", "rope": (0.2, 0.5)}],
                },
            },
            1,
        ),
    ],
)
def test_plot_forest(models, model_fits, args_expected):
    obj = [getattr(models, model_fit) for model_fit in model_fits]
    args, expected = args_expected
    axes = plot_forest(obj, **args)
    assert axes.size == expected


def test_plot_forest_rope_exception():
    with pytest.raises(ValueError) as err:
        plot_forest({"x": [1]}, rope="not_correct_format")
    assert "Argument `rope` must be None, a dictionary like" in str(err.value)


def test_plot_forest_single_value():
    axes = plot_forest({"x": [1]})
    assert axes.shape


def test_plot_forest_ridge_discrete(discrete_model):
    axes = plot_forest(discrete_model, kind="ridgeplot")
    assert axes.shape


@pytest.mark.parametrize("model_fits", [["model_1"], ["model_1", "model_2"]])
def test_plot_forest_bad(models, model_fits):
    obj = [getattr(models, model_fit) for model_fit in model_fits]
    with pytest.raises(TypeError):
        plot_forest(obj, kind="bad_kind")

    with pytest.raises(ValueError):
        plot_forest(obj, model_names=[f"model_name_{i}" for i in range(len(obj) + 10)])


@pytest.mark.parametrize("kind", ["kde", "hist"])
def test_plot_energy(models, kind):
    assert plot_energy(models.model_1, kind=kind)


def test_plot_energy_bad(models):
    with pytest.raises(ValueError):
        plot_energy(models.model_1, kind="bad_kind")


def test_plot_energy_correctly_transposed():
    idata = load_arviz_data("centered_eight")
    idata.sample_stats["energy"] = idata.sample_stats.energy.T
    ax = plot_energy(idata)
    # legend has one entry for each KDE and 1 BFMI for each chain
    assert len(ax.legend_.texts) == 2 + len(idata.sample_stats.chain)


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
        {"is_circular": False},
        {"is_circular": True},
        {"is_circular": "radians"},
        {"is_circular": "degrees"},
        {"adaptive": True},
        {"hdi_probs": [0.3, 0.9, 0.6]},
        {"hdi_probs": [0.3, 0.6, 0.9], "contourf_kwargs": {"cmap": "Blues"}},
        {"hdi_probs": [0.9, 0.6, 0.3], "contour_kwargs": {"alpha": 0}},
    ],
)
def test_plot_kde(continuous_model, kwargs):
    axes = plot_kde(continuous_model["x"], continuous_model["y"], **kwargs)
    axes1 = plot_kde(continuous_model["x"], continuous_model["y"], **kwargs)
    assert axes
    assert axes is axes1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hdi_probs": [1, 2, 3]},
        {"hdi_probs": [-0.3, 0.6, 0.9]},
        {"hdi_probs": [0, 0.3, 0.6]},
        {"hdi_probs": [0.3, 0.6, 1]},
    ],
)
def test_plot_kde_hdi_probs_bad(continuous_model, kwargs):
    """Ensure invalid hdi probabilities are rejected."""
    with pytest.raises(ValueError):
        plot_kde(continuous_model["x"], continuous_model["y"], **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hdi_probs": [0.3, 0.6, 0.9], "contourf_kwargs": {"levels": [0, 0.5, 1]}},
        {"hdi_probs": [0.3, 0.6, 0.9], "contour_kwargs": {"levels": [0, 0.5, 1]}},
    ],
)
def test_plot_kde_hdi_probs_warning(continuous_model, kwargs):
    """Ensure warning is raised when too many keywords are specified."""
    with pytest.warns(UserWarning):
        axes = plot_kde(continuous_model["x"], continuous_model["y"], **kwargs)
    assert axes


@pytest.mark.parametrize("shape", [(8,), (8, 8), (8, 8, 8)])
def test_cov(shape):
    x = np.random.randn(*shape)
    if x.ndim <= 2:
        assert np.allclose(_cov(x), np.cov(x))
    else:
        with pytest.raises(ValueError):
            _cov(x)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cumulative": True},
        {"cumulative": True, "plot_kwargs": {"linestyle": "--"}},
        {"rug": True},
        {"rug": True, "rug_kwargs": {"alpha": 0.2}, "rotated": True},
    ],
)
def test_plot_kde_cumulative(continuous_model, kwargs):
    axes = plot_kde(continuous_model["x"], quantiles=[0.25, 0.5, 0.75], **kwargs)
    assert axes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "hist"},
        {"kind": "kde"},
        {"is_circular": False},
        {"is_circular": False, "kind": "hist"},
        {"is_circular": True},
        {"is_circular": True, "kind": "hist"},
        {"is_circular": "radians"},
        {"is_circular": "radians", "kind": "hist"},
        {"is_circular": "degrees"},
        {"is_circular": "degrees", "kind": "hist"},
    ],
)
def test_plot_dist(continuous_model, kwargs):
    axes = plot_dist(continuous_model["x"], **kwargs)
    axes1 = plot_dist(continuous_model["x"], **kwargs)
    assert axes
    assert axes is axes1


def test_plot_dist_hist(data_random):
    axes = plot_dist(data_random, hist_kwargs=dict(bins=30))
    assert axes


def test_list_conversion(data_list):
    axes = plot_dist(data_list, hist_kwargs=dict(bins=30))
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
    axes = plot_kde(continuous_model["x"], quantiles=[0.05, 0.5, 0.95], **kwargs)
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "var_names": "theta",
            "divergences": True,
            "coords": {"school": [0, 1]},
            "scatter_kwargs": {"marker": "x", "c": "C0"},
            "divergences_kwargs": {"marker": "*", "c": "C0"},
        },
        {
            "divergences": True,
            "scatter_kwargs": {"marker": "x", "c": "C0"},
            "divergences_kwargs": {"marker": "*", "c": "C0"},
            "var_names": ["theta", "mu"],
        },
        {"kind": "kde", "var_names": ["theta"]},
        {"kind": "hexbin", "colorbar": False, "var_names": ["theta"]},
        {"kind": "hexbin", "colorbar": True, "var_names": ["theta"]},
        {
            "kind": "hexbin",
            "var_names": ["theta"],
            "coords": {"school": [0, 1]},
            "colorbar": True,
            "hexbin_kwargs": {"cmap": "viridis"},
            "textsize": 20,
        },
        {
            "point_estimate": "mean",
            "reference_values": {"mu": 0, "tau": 0},
            "reference_values_kwargs": {"c": "C0", "marker": "*"},
        },
        {
            "var_names": ["mu", "tau"],
            "reference_values": {"mu": 0, "tau": 0},
            "labeller": MapLabeller({"mu": r"$\mu$", "theta": r"$\theta"}),
        },
        {
            "var_names": ["theta"],
            "reference_values": {"theta": [0.0] * 8},
            "labeller": MapLabeller({"theta": r"$\theta$"}),
        },
        {
            "var_names": ["theta"],
            "reference_values": {"theta": np.zeros(8)},
            "labeller": MapLabeller({"theta": r"$\theta$"}),
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
    with pytest.warns(UserWarning):
        ax = plot_pair(data, divergences=True)
    assert np.all(ax)


@pytest.mark.parametrize(
    "kwargs", [{}, {"marginals": True}, {"marginals": True, "var_names": ["mu", "tau"]}]
)
def test_plot_pair_overlaid(models, kwargs):
    ax = plot_pair(models.model_1, **kwargs)
    ax2 = plot_pair(models.model_2, ax=ax, **kwargs)
    assert ax is ax2
    assert ax.shape


@pytest.mark.parametrize("marginals", [True, False])
def test_plot_pair_combinedims(models, marginals):
    ax = plot_pair(
        models.model_1, var_names=["eta", "theta"], combine_dims={"school"}, marginals=marginals
    )
    if marginals:
        assert ax.shape == (2, 2)
    else:
        assert not isinstance(ax, np.ndarray)


@pytest.mark.parametrize("marginals", [True, False])
@pytest.mark.parametrize("max_subplots", [True, False])
def test_plot_pair_shapes(marginals, max_subplots):
    rng = np.random.default_rng()
    idata = from_dict({"a": rng.standard_normal((4, 500, 5))})
    if max_subplots:
        with rc_context({"plot.max_subplots": 6}):
            with pytest.warns(UserWarning, match="3x3 grid"):
                ax = plot_pair(idata, marginals=marginals)
    else:
        ax = plot_pair(idata, marginals=marginals)
    side = 3 if max_subplots else (4 + marginals)
    assert ax.shape == (side, side)


@pytest.mark.parametrize("sharex", ["col", None])
@pytest.mark.parametrize("sharey", ["row", None])
@pytest.mark.parametrize("marginals", [True, False])
def test_plot_pair_shared(sharex, sharey, marginals):
    # Generate fake data and plot
    rng = np.random.default_rng()
    idata = from_dict({"a": rng.standard_normal((4, 500, 5))})
    numvars = 5 - (not marginals)
    if sharex is None and sharey is None:
        ax = plot_pair(idata, marginals=marginals)
    else:
        backend_kwargs = {}
        if sharex is not None:
            backend_kwargs["sharex"] = sharex
        if sharey is not None:
            backend_kwargs["sharey"] = sharey
        with pytest.warns(UserWarning):
            ax = plot_pair(idata, marginals=marginals, backend_kwargs=backend_kwargs)

    # Check x axes shared correctly
    for i in range(numvars):
        num_shared_x = numvars - i
        assert len(ax[-1, i].get_shared_x_axes().get_siblings(ax[-1, i])) == num_shared_x

    # Check y axes shared correctly
    for j in range(numvars):
        if marginals:
            num_shared_y = j

            # Check diagonal has unshared axis
            assert len(ax[j, j].get_shared_y_axes().get_siblings(ax[j, j])) == 1

            if j == 0:
                continue
        else:
            num_shared_y = j + 1
        assert len(ax[j, 0].get_shared_y_axes().get_siblings(ax[j, 0])) == num_shared_y


@pytest.mark.parametrize("kind", ["kde", "cumulative", "scatter"])
@pytest.mark.parametrize("alpha", [None, 0.2, 1])
@pytest.mark.parametrize("animated", [False, True])
@pytest.mark.parametrize("observed", [True, False])
@pytest.mark.parametrize("observed_rug", [False, True])
def test_plot_ppc(models, kind, alpha, animated, observed, observed_rug):
    if animation and not animation.writers.is_available("ffmpeg"):
        pytest.skip("matplotlib animations within ArviZ require ffmpeg")
    animation_kwargs = {"blit": False}
    axes = plot_ppc(
        models.model_1,
        kind=kind,
        alpha=alpha,
        observed=observed,
        observed_rug=observed_rug,
        animated=animated,
        animation_kwargs=animation_kwargs,
        random_seed=3,
    )
    if animated:
        assert axes[0]
        assert axes[1]
    assert axes


def test_plot_ppc_transposed():
    idata = load_arviz_data("rugby")
    idata.map(
        lambda ds: ds.assign(points=xr.concat((ds.home_points, ds.away_points), "field")),
        groups="observed_vars",
        inplace=True,
    )
    assert idata.posterior_predictive.points.dims == ("field", "chain", "draw", "match")
    ax = plot_ppc(
        idata,
        kind="scatter",
        var_names="points",
        flatten=["field"],
        coords={"match": ["Wales Italy"]},
        random_seed=3,
        num_pp_samples=8,
    )
    x, y = ax.get_lines()[2].get_data()
    assert not np.isclose(y[0], 0)
    assert np.all(np.array([47, 44, 15, 11]) == x)


@pytest.mark.parametrize("kind", ["kde", "cumulative", "scatter"])
@pytest.mark.parametrize("jitter", [None, 0, 0.1, 1, 3])
@pytest.mark.parametrize("animated", [False, True])
def test_plot_ppc_multichain(kind, jitter, animated):
    if animation and not animation.writers.is_available("ffmpeg"):
        pytest.skip("matplotlib animations within ArviZ require ffmpeg")
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


@pytest.mark.parametrize("kind", ["kde", "cumulative", "scatter"])
@pytest.mark.parametrize("animated", [False, True])
def test_plot_ppc_discrete(kind, animated):
    if animation and not animation.writers.is_available("ffmpeg"):
        pytest.skip("matplotlib animations within ArviZ require ffmpeg")
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


@pytest.mark.skipif(
    not animation.writers.is_available("ffmpeg"),
    reason="matplotlib animations within ArviZ require ffmpeg",
)
@pytest.mark.parametrize("kind", ["kde", "cumulative", "scatter"])
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
    animations_folder = "../saved_animations"
    os.makedirs(animations_folder, exist_ok=True)
    path = os.path.join(animations_folder, f"ppc_{kind}_animation.mp4")
    anim.save(path)
    assert os.path.exists(path)
    assert os.path.getsize(path)


@pytest.mark.skipif(
    not animation.writers.is_available("ffmpeg"),
    reason="matplotlib animations within ArviZ require ffmpeg",
)
@pytest.mark.parametrize("kind", ["kde", "cumulative", "scatter"])
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
    animations_folder = "../saved_animations"
    os.makedirs(animations_folder, exist_ok=True)
    path = os.path.join(animations_folder, f"ppc_discrete_{kind}_animation.mp4")
    anim.save(path)
    assert os.path.exists(path)
    assert os.path.getsize(path)


@pytest.mark.skipif(
    not animation.writers.is_available("ffmpeg"),
    reason="matplotlib animations within ArviZ require ffmpeg",
)
@pytest.mark.parametrize("system", ["Windows", "Darwin"])
def test_non_linux_blit(models, monkeypatch, system, caplog):
    import platform

    def mock_system():
        return system

    monkeypatch.setattr(platform, "system", mock_system)

    animation_kwargs = {"blit": True}
    axes, anim = plot_ppc(
        models.model_1,
        kind="kde",
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"flatten": []},
        {"flatten": [], "coords": {"obs_dim": [1, 2, 3]}},
        {"flatten": ["obs_dim"], "coords": {"obs_dim": [1, 2, 3]}},
    ],
)
def test_plot_ppc_grid(models, kwargs):
    axes = plot_ppc(models.model_1, kind="scatter", **kwargs)
    if not kwargs.get("flatten") and not kwargs.get("coords"):
        assert axes.size == 8
    elif not kwargs.get("flatten"):
        assert axes.size == 3
    else:
        assert not isinstance(axes, np.ndarray)
        assert np.ravel(axes).size == 1


@pytest.mark.parametrize("kind", ["kde", "cumulative", "scatter"])
def test_plot_ppc_bad(models, kind):
    data = from_dict(posterior={"mu": np.random.randn()})
    with pytest.raises(TypeError):
        plot_ppc(data, kind=kind)
    with pytest.raises(TypeError):
        plot_ppc(models.model_1, kind="bad_val")
    with pytest.raises(TypeError):
        plot_ppc(models.model_1, num_pp_samples="bad_val")


@pytest.mark.parametrize("kind", ["kde", "cumulative", "scatter"])
def test_plot_ppc_ax(models, kind, fig_ax):
    """Test ax argument of plot_ppc."""
    _, ax = fig_ax
    axes = plot_ppc(models.model_1, kind=kind, ax=ax)
    assert np.asarray(axes).item(0) is ax


@pytest.mark.skipif(
    not animation.writers.is_available("ffmpeg"),
    reason="matplotlib animations within ArviZ require ffmpeg",
)
def test_plot_ppc_bad_ax(models, fig_ax):
    _, ax = fig_ax
    _, ax2 = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="same figure"):
        plot_ppc(
            models.model_1, ax=[ax, *ax2], flatten=[], coords={"obs_dim": [1, 2, 3]}, animated=True
        )
    with pytest.raises(ValueError, match="2 axes"):
        plot_ppc(models.model_1, ax=ax2)


def test_plot_legend(models):
    axes = plot_ppc(models.model_1)
    legend_texts = axes.get_legend().get_texts()
    result = [i.get_text() for i in legend_texts]
    expected = ["Posterior predictive", "Observed", "Posterior predictive mean"]
    assert result == expected


@pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
@pytest.mark.parametrize("side", ["both", "left", "right"])
@pytest.mark.parametrize("rug", [True])
def test_plot_violin(models, var_names, side, rug):
    axes = plot_violin(models.model_1, var_names=var_names, side=side, rug=rug)
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


@pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
def test_plot_violin_combinedims(models, var_names):
    axes = plot_violin(models.model_1, var_names=var_names, combine_dims={"school"})
    assert axes.shape


def test_plot_violin_ax_combinedims(models):
    _, ax = plt.subplots(1)
    axes = plot_violin(models.model_1, var_names="mu", combine_dims={"school"}, ax=ax)
    assert axes.shape


def test_plot_violin_layout_combinedims(models):
    axes = plot_violin(
        models.model_1, var_names=["mu", "tau"], combine_dims={"school"}, sharey=False
    )
    assert axes.shape


def test_plot_violin_discrete_combinedims(discrete_model):
    axes = plot_violin(discrete_model, combine_dims={"school"})
    assert axes.shape


def test_plot_autocorr_short_chain():
    """Check that logic for small chain defaulting doesn't cause exception"""
    chain = np.arange(10)
    axes = plot_autocorr(chain)
    assert axes


def test_plot_autocorr_uncombined(models):
    axes = plot_autocorr(models.model_1, combined=False)
    assert axes.size
    max_subplots = (
        np.inf if rcParams["plot.max_subplots"] is None else rcParams["plot.max_subplots"]
    )
    assert axes.size == min(72, max_subplots)


def test_plot_autocorr_combined(models):
    axes = plot_autocorr(models.model_1, combined=True)
    assert axes.size == 18


@pytest.mark.parametrize("var_names", (None, "mu", ["mu"], ["mu", "tau"]))
def test_plot_autocorr_var_names(models, var_names):
    axes = plot_autocorr(models.model_1, var_names=var_names, combined=True)
    if (isinstance(var_names, list) and len(var_names) == 1) or isinstance(var_names, str):
        assert not isinstance(axes, np.ndarray)
    else:
        assert axes.shape


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_names": "mu"},
        {"var_names": ("mu", "tau"), "coords": {"school": [0, 1]}},
        {"var_names": "mu", "ref_line": True},
        {
            "var_names": "mu",
            "ref_line_kwargs": {"lw": 2, "color": "C2"},
            "bar_kwargs": {"width": 0.7},
        },
        {"var_names": "mu", "ref_line": False},
        {"var_names": "mu", "kind": "vlines"},
        {
            "var_names": "mu",
            "kind": "vlines",
            "vlines_kwargs": {"lw": 0},
            "marker_vlines_kwargs": {"lw": 3},
        },
    ],
)
def test_plot_rank(models, kwargs):
    axes = plot_rank(models.model_1, **kwargs)
    var_names = kwargs.get("var_names", [])
    if isinstance(var_names, str):
        assert not isinstance(axes, np.ndarray)
    else:
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
        {"hdi_prob": "hide", "label": ""},
        {"point_estimate": None},
        {"ref_val": 0},
        {"ref_val": None},
        {"ref_val": {"mu": [{"ref_val": 1}]}},
        {"bins": None, "kind": "hist"},
        {
            "ref_val": {
                "theta": [
                    # {"school": ["Choate", "Deerfield"], "ref_val": -1}, this is not working
                    {"school": "Lawrenceville", "ref_val": 3}
                ]
            }
        },
    ],
)
def test_plot_posterior(models, kwargs):
    axes = plot_posterior(models.model_1, **kwargs)
    if isinstance(kwargs.get("var_names"), str):
        assert not isinstance(axes, np.ndarray)
    else:
        assert axes.shape


def test_plot_posterior_boolean():
    data = np.random.choice(a=[False, True], size=(4, 100))
    axes = plot_posterior(data)
    assert axes
    plt.draw()
    labels = [label.get_text() for label in axes.get_xticklabels()]
    assert all(item in labels for item in ("True", "False"))


@pytest.mark.parametrize("kwargs", [{}, {"point_estimate": "mode"}, {"bins": None, "kind": "hist"}])
def test_plot_posterior_discrete(discrete_model, kwargs):
    axes = plot_posterior(discrete_model, **kwargs)
    assert axes.shape


def test_plot_posterior_bad_type():
    with pytest.raises(TypeError):
        plot_posterior(np.array(["a", "b", "c"]))


def test_plot_posterior_bad(models):
    with pytest.raises(ValueError):
        plot_posterior(models.model_1, rope="bad_value")
    with pytest.raises(ValueError):
        plot_posterior(models.model_1, ref_val="bad_value")
    with pytest.raises(ValueError):
        plot_posterior(models.model_1, point_estimate="bad_value")


@pytest.mark.parametrize("point_estimate", ("mode", "mean", "median"))
def test_plot_posterior_point_estimates(models, point_estimate):
    axes = plot_posterior(models.model_1, var_names=("mu", "tau"), point_estimate=point_estimate)
    assert axes.size == 2


def test_plot_posterior_skipna():
    sample = np.linspace(0, 1)
    sample[:10] = np.nan
    plot_posterior({"a": sample}, skipna=True)
    with pytest.raises(ValueError):
        plot_posterior({"a": sample}, skipna=False)


@pytest.mark.parametrize("kwargs", [{"var_names": ["mu", "theta"]}])
def test_plot_posterior_combinedims(models, kwargs):
    axes = plot_posterior(models.model_1, combine_dims={"school"}, **kwargs)
    if isinstance(kwargs.get("var_names"), str):
        assert not isinstance(axes, np.ndarray)
    else:
        assert axes.shape


@pytest.mark.parametrize("kwargs", [{}, {"point_estimate": "mode"}, {"bins": None, "kind": "hist"}])
def test_plot_posterior_discrete_combinedims(discrete_multidim_model, kwargs):
    axes = plot_posterior(discrete_multidim_model, combine_dims={"school"}, **kwargs)
    assert axes.size == 2


@pytest.mark.parametrize("point_estimate", ("mode", "mean", "median"))
def test_plot_posterior_point_estimates_combinedims(models, point_estimate):
    axes = plot_posterior(
        models.model_1,
        var_names=("mu", "tau"),
        combine_dims={"school"},
        point_estimate=point_estimate,
    )
    assert axes.size == 2


def test_plot_posterior_skipna_combinedims():
    idata = load_arviz_data("centered_eight")
    idata.posterior["theta"].loc[dict(school="Deerfield")] = np.nan
    with pytest.raises(ValueError):
        plot_posterior(idata, var_names="theta", combine_dims={"school"}, skipna=False)
    ax = plot_posterior(idata, var_names="theta", combine_dims={"school"}, skipna=True)
    assert not isinstance(ax, np.ndarray)


@pytest.mark.parametrize(
    "kwargs", [{"insample_dev": True}, {"plot_standard_error": False}, {"plot_ic_diff": False}]
)
def test_plot_compare(models, kwargs):
    model_compare = compare({"Model 1": models.model_1, "Model 2": models.model_2})

    axes = plot_compare(model_compare, **kwargs)
    assert axes


def test_plot_compare_no_ic(models):
    """Check exception is raised if model_compare doesn't contain a valid information criterion"""
    model_compare = compare({"Model 1": models.model_1, "Model 2": models.model_2})

    # Drop column needed for plotting
    model_compare = model_compare.drop("elpd_loo", axis=1)
    with pytest.raises(ValueError) as err:
        plot_compare(model_compare)

    assert "comp_df must contain one of the following" in str(err.value)
    assert "['elpd_loo', 'elpd_waic']" in str(err.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"color": "0.5", "circular": True},
        {"hdi_data": True, "fill_kwargs": {"alpha": 0}},
        {"plot_kwargs": {"alpha": 0}},
        {"smooth_kwargs": {"window_length": 33, "polyorder": 5, "mode": "mirror"}},
        {"hdi_data": True, "smooth": False},
    ],
)
def test_plot_hdi(models, data, kwargs):
    hdi_data = kwargs.pop("hdi_data", None)
    if hdi_data:
        hdi_data = hdi(models.model_1.posterior["theta"])
        ax = plot_hdi(data["y"], hdi_data=hdi_data, **kwargs)
    else:
        ax = plot_hdi(data["y"], models.model_1.posterior["theta"], **kwargs)
    assert ax


def test_plot_hdi_warning():
    """Check using both y and hdi_data sends a warning."""
    x_data = np.random.normal(0, 1, 100)
    y_data = np.random.normal(2 + x_data * 0.5, 0.5, (1, 200, 100))
    hdi_data = hdi(y_data)
    with pytest.warns(UserWarning, match="Both y and hdi_data"):
        ax = plot_hdi(x_data, y=y_data, hdi_data=hdi_data)
    assert ax


def test_plot_hdi_missing_arg_error():
    """Check that both y and hdi_data missing raises an error."""
    with pytest.raises(ValueError, match="One of {y, hdi_data"):
        plot_hdi(np.arange(20))


def test_plot_hdi_dataset_error(models):
    """Check hdi_data as multiple variable Dataset raises an error."""
    hdi_data = hdi(models.model_1)
    with pytest.raises(ValueError, match="Only single variable Dataset"):
        plot_hdi(np.arange(8), hdi_data=hdi_data)


def test_plot_hdi_string_error():
    """Check x as type string raises an error."""
    x_data = ["a", "b", "c", "d"]
    y_data = np.random.normal(0, 5, (1, 200, len(x_data)))
    hdi_data = hdi(y_data)
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            (
                "The `arviz.plot_hdi()` function does not support categorical data. "
                "Consider using `arviz.plot_forest()`."
            )
        ),
    ):
        plot_hdi(x=x_data, y=y_data, hdi_data=hdi_data)


def test_plot_hdi_datetime_error():
    """Check x as datetime raises an error."""
    x_data = np.arange(start="2022-01-01", stop="2022-03-01", dtype=np.datetime64)
    y_data = np.random.normal(0, 5, (1, 200, x_data.shape[0]))
    hdi_data = hdi(y_data)
    with pytest.raises(TypeError, match="Cannot deal with x as type datetime."):
        plot_hdi(x=x_data, y=y_data, hdi_data=hdi_data)


@pytest.mark.parametrize("limits", [(-10.0, 10.0), (-5, 5), (None, None)])
def test_kde_scipy(limits):
    """
    Evaluates if sum of density is the same for our implementation
    and the implementation in scipy
    """
    data = np.random.normal(0, 1, 10000)
    grid, density_own = _kde(data, custom_lims=limits)
    density_sp = gaussian_kde(data).evaluate(grid)
    np.testing.assert_almost_equal(density_own.sum(), density_sp.sum(), 1)


@pytest.mark.parametrize("limits", [(-10.0, 10.0), (-5, 5), (None, None)])
def test_kde_cumulative(limits):
    """
    Evaluates if last value of cumulative density is 1
    """
    data = np.random.normal(0, 1, 1000)
    density = _kde(data, custom_lims=limits, cumulative=True)[1]
    np.testing.assert_almost_equal(round(density[-1], 3), 1)


def test_plot_ecdf_basic():
    data = np.random.randn(4, 1000)
    axes = plot_ecdf(data)
    assert axes is not None


def test_plot_ecdf_eval_points():
    """Check that BehaviourChangeWarning is raised if eval_points is not specified."""
    data = np.random.randn(4, 1000)
    eval_points = np.linspace(-3, 3, 100)
    with pytest.warns(BehaviourChangeWarning):
        axes = plot_ecdf(data)
    assert axes is not None
    with does_not_warn(BehaviourChangeWarning):
        axes = plot_ecdf(data, eval_points=eval_points)
    assert axes is not None


@pytest.mark.parametrize("confidence_bands", [True, "pointwise", "optimized", "simulated"])
@pytest.mark.parametrize("ndraws", [100, 10_000])
def test_plot_ecdf_confidence_bands(confidence_bands, ndraws):
    """Check that all confidence_bands values correctly accepted"""
    data = np.random.randn(4, ndraws // 4)
    axes = plot_ecdf(data, confidence_bands=confidence_bands, cdf=norm(0, 1).cdf)
    assert axes is not None


def test_plot_ecdf_values2():
    data = np.random.randn(4, 1000)
    data2 = np.random.randn(4, 1000)
    axes = plot_ecdf(data, data2)
    assert axes is not None


def test_plot_ecdf_cdf():
    data = np.random.randn(4, 1000)
    cdf = norm(0, 1).cdf
    axes = plot_ecdf(data, cdf=cdf)
    assert axes is not None


def test_plot_ecdf_error():
    """Check that all error conditions are correctly raised."""
    dist = norm(0, 1)
    data = dist.rvs(1000)

    # cdf not specified
    with pytest.raises(ValueError):
        plot_ecdf(data, confidence_bands=True)
    plot_ecdf(data, confidence_bands=True, cdf=dist.cdf)
    with pytest.raises(ValueError):
        plot_ecdf(data, difference=True)
    plot_ecdf(data, difference=True, cdf=dist.cdf)
    with pytest.raises(ValueError):
        plot_ecdf(data, pit=True)
    plot_ecdf(data, pit=True, cdf=dist.cdf)

    # contradictory confidence band types
    with pytest.raises(ValueError):
        plot_ecdf(data, cdf=dist.cdf, confidence_bands="simulated", pointwise=True)
    with pytest.raises(ValueError):
        plot_ecdf(data, cdf=dist.cdf, confidence_bands="optimized", pointwise=True)
    plot_ecdf(data, cdf=dist.cdf, confidence_bands=True, pointwise=True)
    plot_ecdf(data, cdf=dist.cdf, confidence_bands="pointwise")

    # contradictory band probabilities
    with pytest.raises(ValueError):
        plot_ecdf(data, cdf=dist.cdf, confidence_bands=True, ci_prob=0.9, fpr=0.1)
    plot_ecdf(data, cdf=dist.cdf, confidence_bands=True, ci_prob=0.9)
    plot_ecdf(data, cdf=dist.cdf, confidence_bands=True, fpr=0.1)

    # contradictory reference
    data2 = dist.rvs(200)
    with pytest.raises(ValueError):
        plot_ecdf(data, data2, cdf=dist.cdf, difference=True)
    plot_ecdf(data, data2, difference=True)
    plot_ecdf(data, cdf=dist.cdf, difference=True)


def test_plot_ecdf_deprecations():
    """Check that deprecations are raised correctly."""
    dist = norm(0, 1)
    data = dist.rvs(1000)
    # base case, no deprecations
    with does_not_warn(FutureWarning):
        axes = plot_ecdf(data, cdf=dist.cdf, confidence_bands=True)
    assert axes is not None

    # values2 is deprecated
    data2 = dist.rvs(200)
    with pytest.warns(FutureWarning):
        axes = plot_ecdf(data, values2=data2, difference=True)

    # pit is deprecated
    with pytest.warns(FutureWarning):
        axes = plot_ecdf(data, cdf=dist.cdf, pit=True)
    assert axes is not None

    # fpr is deprecated
    with does_not_warn(FutureWarning):
        axes = plot_ecdf(data, cdf=dist.cdf, ci_prob=0.9)
    with pytest.warns(FutureWarning):
        axes = plot_ecdf(data, cdf=dist.cdf, confidence_bands=True, fpr=0.1)
    assert axes is not None

    # pointwise is deprecated
    with does_not_warn(FutureWarning):
        axes = plot_ecdf(data, cdf=dist.cdf, confidence_bands="pointwise")
    with pytest.warns(FutureWarning):
        axes = plot_ecdf(data, cdf=dist.cdf, confidence_bands=True, pointwise=True)


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
        {"threshold": 2},
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
    with pytest.raises(ValueError):
        plot_elpd(model_dict)


def test_plot_elpd_scale_error(models):
    model_dict = {
        "Model 1": waic(models.model_1, pointwise=True, scale="log"),
        "Model 2": waic(models.model_2, pointwise=True, scale="deviance"),
    }
    with pytest.raises(ValueError):
        plot_elpd(model_dict)


def test_plot_elpd_one_model(models):
    model_dict = {"Model 1": models.model_1}
    with pytest.raises(Exception):
        plot_elpd(model_dict)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"xlabels": True},
        {"color": "obs_dim", "xlabels": True, "show_bins": True, "bin_format": "{0}"},
        {"color": "obs_dim", "legend": True, "hover_label": True},
        {"color": "blue", "coords": {"obs_dim": slice(2, 4)}},
        {"color": np.random.uniform(size=8), "show_bins": True},
        {
            "color": np.random.uniform(size=(8, 3)),
            "show_bins": True,
            "show_hlines": True,
            "threshold": 1,
        },
    ],
)
@pytest.mark.parametrize("input_type", ["elpd_data", "data_array", "array"])
def test_plot_khat(models, input_type, kwargs):
    khats_data = loo(models.model_1, pointwise=True)

    if input_type == "data_array":
        khats_data = khats_data.pareto_k
    elif input_type == "array":
        khats_data = khats_data.pareto_k.values
        if "color" in kwargs and isinstance(kwargs["color"], str) and kwargs["color"] == "obs_dim":
            kwargs["color"] = None

    axes = plot_khat(khats_data, **kwargs)
    assert axes


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"xlabels": True},
        {"color": "dim1", "xlabels": True, "show_bins": True, "bin_format": "{0}"},
        {"color": "dim2", "legend": True, "hover_label": True},
        {"color": "blue", "coords": {"dim2": slice(2, 4)}},
        {"color": np.random.uniform(size=35), "show_bins": True},
        {
            "color": np.random.uniform(size=(35, 3)),
            "show_bins": True,
            "show_hlines": True,
            "threshold": 1,
        },
    ],
)
@pytest.mark.parametrize("input_type", ["elpd_data", "data_array", "array"])
def test_plot_khat_multidim(multidim_models, input_type, kwargs):
    khats_data = loo(multidim_models.model_1, pointwise=True)

    if input_type == "data_array":
        khats_data = khats_data.pareto_k
    elif input_type == "array":
        khats_data = khats_data.pareto_k.values
        if (
            "color" in kwargs
            and isinstance(kwargs["color"], str)
            and kwargs["color"] in ("dim1", "dim2")
        ):
            kwargs["color"] = None

    axes = plot_khat(khats_data, **kwargs)
    assert axes


def test_plot_khat_threshold():
    khats = np.array([0, 0, 0.6, 0.6, 0.8, 0.9, 0.9, 2, 3, 4, 1.5])
    axes = plot_khat(khats, threshold=1)
    assert axes


def test_plot_khat_bad_input(models):
    with pytest.raises(ValueError):
        plot_khat(models.model_1.sample_stats)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_names": ["theta"], "relative": True, "color": "r"},
        {"coords": {"school": slice(4)}, "n_points": 10},
        {"min_ess": 600, "hline_kwargs": {"color": "r"}},
    ],
)
@pytest.mark.parametrize("kind", ["local", "quantile", "evolution"])
def test_plot_ess(models, kind, kwargs):
    """Test plot_ess arguments common to all kind of plots."""
    idata = models.model_1
    ax = plot_ess(idata, kind=kind, **kwargs)
    assert np.all(ax)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rug": True},
        {"rug": True, "rug_kind": "max_depth", "rug_kwargs": {"color": "c"}},
        {"extra_methods": True},
        {"extra_methods": True, "extra_kwargs": {"ls": ":"}, "text_kwargs": {"x": 0, "ha": "left"}},
        {"extra_methods": True, "rug": True},
    ],
)
@pytest.mark.parametrize("kind", ["local", "quantile"])
def test_plot_ess_local_quantile(models, kind, kwargs):
    """Test specific arguments in kinds local and quantile of plot_ess."""
    idata = models.model_1
    ax = plot_ess(idata, kind=kind, **kwargs)
    assert np.all(ax)


def test_plot_ess_evolution(models):
    """Test specific arguments in evolution kind of plot_ess."""
    idata = models.model_1
    ax = plot_ess(idata, kind="evolution", extra_kwargs={"linestyle": "--"}, color="b")
    assert np.all(ax)


def test_plot_ess_bad_kind(models):
    """Test error when plot_ess receives an invalid kind."""
    idata = models.model_1
    with pytest.raises(ValueError, match="Invalid kind"):
        plot_ess(idata, kind="bad kind")


@pytest.mark.parametrize("dim", ["chain", "draw"])
def test_plot_ess_bad_coords(models, dim):
    """Test error when chain or dim are used as coords to select a data subset."""
    idata = models.model_1
    with pytest.raises(ValueError, match="invalid coordinates"):
        plot_ess(idata, coords={dim: slice(3)})


def test_plot_ess_no_sample_stats(models):
    """Test error when rug=True but sample_stats group is not present."""
    idata = models.model_1
    with pytest.raises(ValueError, match="must contain sample_stats"):
        plot_ess(idata.posterior, rug=True)


def test_plot_ess_no_divergences(models):
    """Test error when rug=True, but the variable defined by rug_kind is missing."""
    idata = deepcopy(models.model_1)
    idata.sample_stats = idata.sample_stats.rename({"diverging": "diverging_missing"})
    with pytest.raises(ValueError, match="not contain diverging"):
        plot_ess(idata, rug=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_unif": 50, "legend": False},
        {"use_hdi": True, "color": "gray"},
        {"use_hdi": True, "hdi_prob": 0.68},
        {"use_hdi": True, "hdi_kwargs": {"fill": 0.1}},
        {"ecdf": True},
        {"ecdf": True, "ecdf_fill": False, "plot_unif_kwargs": {"ls": "--"}},
        {"ecdf": True, "hdi_prob": 0.97, "fill_kwargs": {"hatch": "/"}},
    ],
)
def test_plot_loo_pit(models, kwargs):
    axes = plot_loo_pit(idata=models.model_1, y="y", **kwargs)
    assert axes


def test_plot_loo_pit_incompatible_args(models):
    """Test error when both ecdf and use_hdi are True."""
    with pytest.raises(ValueError, match="incompatible"):
        plot_loo_pit(idata=models.model_1, y="y", ecdf=True, use_hdi=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_names": ["theta"], "color": "r"},
        {"rug": True, "rug_kwargs": {"color": "r"}},
        {"errorbar": True, "rug": True, "rug_kind": "max_depth"},
        {"errorbar": True, "coords": {"school": slice(4)}, "n_points": 10},
        {"extra_methods": True, "rug": True},
        {"extra_methods": True, "extra_kwargs": {"ls": ":"}, "text_kwargs": {"x": 0, "ha": "left"}},
    ],
)
def test_plot_mcse(models, kwargs):
    idata = models.model_1
    ax = plot_mcse(idata, **kwargs)
    assert np.all(ax)


@pytest.mark.parametrize("dim", ["chain", "draw"])
def test_plot_mcse_bad_coords(models, dim):
    """Test error when chain or dim are used as coords to select a data subset."""
    idata = models.model_1
    with pytest.raises(ValueError, match="invalid coordinates"):
        plot_mcse(idata, coords={dim: slice(3)})


def test_plot_mcse_no_sample_stats(models):
    """Test error when rug=True but sample_stats group is not present."""
    idata = models.model_1
    with pytest.raises(ValueError, match="must contain sample_stats"):
        plot_mcse(idata.posterior, rug=True)


def test_plot_mcse_no_divergences(models):
    """Test error when rug=True, but the variable defined by rug_kind is missing."""
    idata = deepcopy(models.model_1)
    idata.sample_stats = idata.sample_stats.rename({"diverging": "diverging_missing"})
    with pytest.raises(ValueError, match="not contain diverging"):
        plot_mcse(idata, rug=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_names": ["theta"]},
        {"var_names": ["theta"], "coords": {"school": [0, 1]}},
        {"var_names": ["eta"], "posterior_kwargs": {"rug": True, "rug_kwargs": {"color": "r"}}},
        {"var_names": ["mu"], "prior_kwargs": {"fill_kwargs": {"alpha": 0.5}}},
        {
            "var_names": ["tau"],
            "prior_kwargs": {"plot_kwargs": {"color": "r"}},
            "posterior_kwargs": {"plot_kwargs": {"color": "b"}},
        },
        {"var_names": ["y"], "kind": "observed"},
    ],
)
def test_plot_dist_comparison(models, kwargs):
    idata = models.model_1
    ax = plot_dist_comparison(idata, **kwargs)
    assert np.all(ax)


def test_plot_dist_comparison_different_vars():
    data = from_dict(
        posterior={
            "x": np.random.randn(4, 100, 30),
        },
        prior={"x_hat": np.random.randn(4, 100, 30)},
    )
    with pytest.raises(KeyError):
        plot_dist_comparison(data, var_names="x")
    ax = plot_dist_comparison(data, var_names=[["x_hat"], ["x"]])
    assert np.all(ax)


def test_plot_dist_comparison_combinedims(models):
    idata = models.model_1
    ax = plot_dist_comparison(idata, combine_dims={"school"})
    assert np.all(ax)


def test_plot_dist_comparison_different_vars_combinedims():
    data = from_dict(
        posterior={
            "x": np.random.randn(4, 100, 30),
        },
        prior={"x_hat": np.random.randn(4, 100, 30)},
        dims={"x": ["3rd_dim"], "x_hat": ["3rd_dim"]},
    )
    with pytest.raises(KeyError):
        plot_dist_comparison(data, var_names="x", combine_dims={"3rd_dim"})
    ax = plot_dist_comparison(data, var_names=[["x_hat"], ["x"]], combine_dims={"3rd_dim"})
    assert np.all(ax)
    assert ax.size == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"reference": "analytical"},
        {"kind": "p_value"},
        {"kind": "t_stat", "t_stat": "std"},
        {"kind": "t_stat", "t_stat": 0.5, "bpv": True},
    ],
)
def test_plot_bpv(models, kwargs):
    axes = plot_bpv(models.model_1, **kwargs)
    assert not isinstance(axes, np.ndarray)


def test_plot_bpv_discrete():
    fake_obs = {"a": np.random.poisson(2.5, 100)}
    fake_pp = {"a": np.random.poisson(2.5, (1, 10, 100))}
    fake_model = from_dict(posterior_predictive=fake_pp, observed_data=fake_obs)
    axes = plot_bpv(fake_model)
    assert not isinstance(axes, np.ndarray)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "binwidth": 0.5,
            "stackratio": 2,
            "nquantiles": 20,
        },
        {"point_interval": True},
        {
            "point_interval": True,
            "dotsize": 1.2,
            "point_estimate": "median",
            "plot_kwargs": {"color": "grey"},
        },
        {
            "point_interval": True,
            "plot_kwargs": {"color": "grey"},
            "nquantiles": 100,
            "hdi_prob": 0.95,
            "intervalcolor": "green",
        },
        {
            "point_interval": True,
            "plot_kwargs": {"color": "grey"},
            "quartiles": False,
            "linewidth": 2,
        },
    ],
)
def test_plot_dot(continuous_model, kwargs):
    data = continuous_model["x"]
    ax = plot_dot(data, **kwargs)
    assert ax


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rotated": True},
        {
            "point_interval": True,
            "rotated": True,
            "dotcolor": "grey",
            "binwidth": 0.5,
        },
        {
            "rotated": True,
            "point_interval": True,
            "plot_kwargs": {"color": "grey"},
            "nquantiles": 100,
            "dotsize": 0.8,
            "hdi_prob": 0.95,
            "intervalcolor": "green",
        },
    ],
)
def test_plot_dot_rotated(continuous_model, kwargs):
    data = continuous_model["x"]
    ax = plot_dot(data, **kwargs)
    assert ax


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "point_estimate": "mean",
            "hdi_prob": 0.95,
            "quartiles": False,
            "linewidth": 2,
            "markersize": 2,
            "markercolor": "red",
            "marker": "o",
            "rotated": False,
            "intervalcolor": "green",
        },
    ],
)
def test_plot_point_interval(continuous_model, kwargs):
    _, ax = plt.subplots()
    data = continuous_model["x"]
    values = np.sort(data)
    ax = plot_point_interval(ax, values, **kwargs)
    assert ax


def test_wilkinson_algorithm(continuous_model):
    data = continuous_model["x"]
    values = np.sort(data)
    _, stack_counts = wilkinson_algorithm(values, 0.5)
    assert np.sum(stack_counts) == len(values)
    stack_locs, stack_counts = wilkinson_algorithm([0.0, 1.0, 1.8, 3.0, 5.0], 1.0)
    assert stack_locs == [0.0, 1.4, 3.0, 5.0]
    assert stack_counts == [1, 2, 1, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"y_hat": "bad_name"},
        {"x": "x1"},
        {"x": ("x1", "x2")},
        {
            "x": ("x1", "x2"),
            "y_kwargs": {"color": "blue", "marker": "^"},
            "y_hat_plot_kwargs": {"color": "cyan"},
        },
        {"x": ("x1", "x2"), "y_model_plot_kwargs": {"color": "red"}},
        {
            "x": ("x1", "x2"),
            "kind_pp": "hdi",
            "kind_model": "hdi",
            "y_model_fill_kwargs": {"color": "red"},
            "y_hat_fill_kwargs": {"color": "cyan"},
        },
    ],
)
def test_plot_lm_1d(models, kwargs):
    """Test functionality for 1D data."""
    idata = models.model_1
    if "constant_data" not in idata.groups():
        y = idata.observed_data["y"]
        x1data = y.coords[y.dims[0]]
        idata.add_groups({"constant_data": {"_": x1data}})
        idata.constant_data["x1"] = x1data
        idata.constant_data["x2"] = x1data

    axes = plot_lm(idata=idata, y="y", y_model="eta", xjitter=True, **kwargs)
    assert np.all(axes)


def test_plot_lm_multidim(multidim_models):
    """Test functionality for multidimentional data."""
    idata = multidim_models.model_1
    axes = plot_lm(
        idata=idata,
        x=idata.observed_data["y"].coords["dim1"].values,
        y="y",
        xjitter=True,
        plot_dim="dim1",
        show=False,
        figsize=(4, 16),
    )
    assert np.all(axes)


@pytest.mark.parametrize(
    "val_err_kwargs",
    [
        {},
        {"kind_pp": "bad_kind"},
        {"kind_model": "bad_kind"},
    ],
)
def test_plot_lm_valueerror(multidim_models, val_err_kwargs):
    """Test error plot_dim gets no value for multidim data and wrong value in kind_... args."""
    idata2 = multidim_models.model_1
    with pytest.raises(ValueError):
        plot_lm(idata=idata2, y="y", **val_err_kwargs)


@pytest.mark.parametrize(
    "warn_kwargs",
    [
        {"y_hat": "bad_name"},
        {"y_model": "bad_name"},
    ],
)
def test_plot_lm_warning(models, warn_kwargs):
    """Test Warning when needed groups or variables are not there in idata."""
    idata1 = models.model_1
    with pytest.warns(UserWarning):
        plot_lm(
            idata=from_dict(observed_data={"y": idata1.observed_data["y"].values}),
            y="y",
            **warn_kwargs,
        )
    with pytest.warns(UserWarning):
        plot_lm(idata=idata1, y="y", **warn_kwargs)


def test_plot_lm_typeerror(models):
    """Test error when invalid value passed to num_samples."""
    idata1 = models.model_1
    with pytest.raises(TypeError):
        plot_lm(idata=idata1, y="y", num_samples=-1)


def test_plot_lm_list():
    """Test the plots when input data is list or ndarray."""
    y = [1, 2, 3, 4, 5]
    assert plot_lm(y=y, x=np.arange(len(y)), show=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"y_hat": "bad_name"},
        {"x": "x"},
        {"x": ("x", "x")},
        {"y_holdout": "z"},
        {"y_holdout": "z", "x_holdout": "x_pred"},
        {"x": ("x", "x"), "y_holdout": "z", "x_holdout": ("x_pred", "x_pred")},
        {"y_forecasts": "z"},
        {"y_holdout": "z", "y_forecasts": "bad_name"},
    ],
)
def test_plot_ts(kwargs):
    """Test timeseries plots basic functionality."""
    nchains = 4
    ndraws = 500
    obs_data = {
        "y": 2 * np.arange(1, 9) + 3,
        "z": 2 * np.arange(8, 12) + 3,
    }

    posterior_predictive = {
        "y": np.random.normal(
            (obs_data["y"] * 1.2) - 3, size=(nchains, ndraws, len(obs_data["y"]))
        ),
        "z": np.random.normal(
            (obs_data["z"] * 1.2) - 3, size=(nchains, ndraws, len(obs_data["z"]))
        ),
    }

    const_data = {"x": np.arange(1, 9), "x_pred": np.arange(8, 12)}

    idata = from_dict(
        observed_data=obs_data,
        posterior_predictive=posterior_predictive,
        constant_data=const_data,
        coords={"obs_dim": np.arange(1, 9), "pred_dim": np.arange(8, 12)},
        dims={"y": ["obs_dim"], "z": ["pred_dim"]},
    )

    ax = plot_ts(idata=idata, y="y", **kwargs)
    assert np.all(ax)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "y_holdout": "z",
            "holdout_dim": "holdout_dim1",
            "x": ("x", "x"),
            "x_holdout": ("x_pred", "x_pred"),
        },
        {"y_forecasts": "z", "holdout_dim": "holdout_dim1"},
    ],
)
def test_plot_ts_multidim(kwargs):
    """Test timeseries plots multidim functionality."""
    nchains = 4
    ndraws = 500
    ndim1 = 5
    ndim2 = 7
    data = {
        "y": np.random.normal(size=(ndim1, ndim2)),
        "z": np.random.normal(size=(ndim1, ndim2)),
    }

    posterior_predictive = {
        "y": np.random.randn(nchains, ndraws, ndim1, ndim2),
        "z": np.random.randn(nchains, ndraws, ndim1, ndim2),
    }

    const_data = {"x": np.arange(1, 6), "x_pred": np.arange(5, 10)}

    idata = from_dict(
        observed_data=data,
        posterior_predictive=posterior_predictive,
        constant_data=const_data,
        dims={
            "y": ["dim1", "dim2"],
            "z": ["holdout_dim1", "holdout_dim2"],
        },
        coords={
            "dim1": range(ndim1),
            "dim2": range(ndim2),
            "holdout_dim1": range(ndim1 - 1, ndim1 + 4),
            "holdout_dim2": range(ndim2 - 1, ndim2 + 6),
        },
    )

    ax = plot_ts(idata=idata, y="y", plot_dim="dim1", **kwargs)
    assert np.all(ax)


@pytest.mark.parametrize("val_err_kwargs", [{}, {"plot_dim": "dim1", "y_holdout": "y"}])
def test_plot_ts_valueerror(multidim_models, val_err_kwargs):
    """Test error plot_dim gets no value for multidim data and wrong value in kind_... args."""
    idata2 = multidim_models.model_1
    with pytest.raises(ValueError):
        plot_ts(idata=idata2, y="y", **val_err_kwargs)


def test_plot_bf():
    idata = from_dict(
        posterior={"a": np.random.normal(1, 0.5, 5000)}, prior={"a": np.random.normal(0, 1, 5000)}
    )
    _, bf_plot = plot_bf(idata, var_name="a", ref_val=0)
    assert bf_plot is not None


def generate_lm_1d_data():
    rng = np.random.default_rng()
    return from_dict(
        observed_data={"y": rng.normal(size=7)},
        posterior_predictive={"y": rng.normal(size=(4, 1000, 7)) / 2},
        posterior={"y_model": rng.normal(size=(4, 1000, 7))},
        dims={"y": ["dim1"]},
        coords={"dim1": range(7)},
    )


def generate_lm_2d_data():
    rng = np.random.default_rng()
    return from_dict(
        observed_data={"y": rng.normal(size=(5, 7))},
        posterior_predictive={"y": rng.normal(size=(4, 1000, 5, 7)) / 2},
        posterior={"y_model": rng.normal(size=(4, 1000, 5, 7))},
        dims={"y": ["dim1", "dim2"]},
        coords={"dim1": range(5), "dim2": range(7)},
    )


@pytest.mark.parametrize("data", ("1d", "2d"))
@pytest.mark.parametrize("kind", ("lines", "hdi"))
@pytest.mark.parametrize("use_y_model", (True, False))
def test_plot_lm(data, kind, use_y_model):
    if data == "1d":
        idata = generate_lm_1d_data()
    else:
        idata = generate_lm_2d_data()

    kwargs = {"idata": idata, "y": "y", "kind_model": kind}
    if data == "2d":
        kwargs["plot_dim"] = "dim1"
    if use_y_model:
        kwargs["y_model"] = "y_model"
    if kind == "lines":
        kwargs["num_samples"] = 50

    ax = plot_lm(**kwargs)
    assert ax is not None


@pytest.mark.parametrize(
    "coords, expected_vars",
    [
        ({"school": ["Choate"]}, ["theta"]),
        ({"school": ["Lawrenceville"]}, ["theta"]),
        ({}, ["theta"]),
    ],
)
def test_plot_autocorr_coords(coords, expected_vars):
    """Test plot_autocorr with coords kwarg."""
    idata = load_arviz_data("centered_eight")

    axes = plot_autocorr(idata, var_names=expected_vars, coords=coords, show=False)
    assert axes is not None


def test_plot_forest_with_transform():
    """Test if plot_forest runs successfully with a transform dictionary."""
    data = xr.Dataset(
        {
            "var1": (["chain", "draw"], np.array([[1, 2, 3], [4, 5, 6]])),
            "var2": (["chain", "draw"], np.array([[7, 8, 9], [10, 11, 12]])),
        },
        coords={"chain": [0, 1], "draw": [0, 1, 2]},
    )
    transform_dict = {
        "var1": lambda x: x + 1,
        "var2": lambda x: x * 2,
    }

    axes = plot_forest(data, transform=transform_dict, show=False)
    assert axes is not None
