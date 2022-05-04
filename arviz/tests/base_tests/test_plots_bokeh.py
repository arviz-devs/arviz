# pylint: disable=redefined-outer-name,too-many-lines
"""Tests use the 'bokeh' backend."""
from copy import deepcopy

import numpy as np
import pytest
from pandas import DataFrame  # pylint: disable=wrong-import-position
from scipy.stats import norm  # pylint: disable=wrong-import-position

from ...data import from_dict, load_arviz_data  # pylint: disable=wrong-import-position
from ...plots import (  # pylint: disable=wrong-import-position
    plot_autocorr,
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
    plot_violin,
)
from ...rcparams import rc_context, rcParams  # pylint: disable=wrong-import-position
from ...stats import compare, hdi, loo, waic  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
    create_model,
    create_multidimensional_model,
    eight_schools_params,
    importorskip,
    models,
    multidim_models,
)

# Skip tests if bokeh not installed
bkp = importorskip("bokeh.plotting")  # pylint: disable=invalid-name


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
        {"point_estimate": "mean"},
        {"point_estimate": "median"},
        {"hdi_prob": 0.94},
        {"hdi_prob": 1},
        {"outline": True},
        {"hdi_markers": ["v"]},
        {"shade": 1},
    ],
)
def test_plot_density_float(models, kwargs):
    obj = [getattr(models, model_fit) for model_fit in ["model_1", "model_2"]]
    axes = plot_density(obj, backend="bokeh", show=False, **kwargs)
    assert axes.shape[0] >= 6
    assert axes.shape[0] >= 3


def test_plot_density_discrete(discrete_model):
    axes = plot_density(discrete_model, shade=0.9, backend="bokeh", show=False)
    assert axes.shape[0] == 1


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
    axes = plot_density([model_ab, model_bc], backend="bokeh", show=False)
    assert axes.size == 3


def test_plot_density_one_var():
    """Test plot_density works when there is only one variable (#1401)."""
    model_ab = from_dict(
        {
            "a": np.random.normal(size=200),
        }
    )
    model_bc = from_dict(
        {
            "a": np.random.normal(size=200),
        }
    )
    axes = plot_density([model_ab, model_bc], backend="bokeh", show=False)
    assert axes.size == 1


def test_plot_density_bad_kwargs(models):
    obj = [getattr(models, model_fit) for model_fit in ["model_1", "model_2"]]
    with pytest.raises(ValueError):
        plot_density(obj, point_estimate="bad_value", backend="bokeh", show=False)

    with pytest.raises(ValueError):
        plot_density(
            obj,
            data_labels=[f"bad_value_{i}" for i in range(len(obj) + 10)],
            backend="bokeh",
            show=False,
        )

    with pytest.raises(ValueError):
        plot_density(obj, hdi_prob=2, backend="bokeh", show=False)


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
    ax = plot_separation(idata=idata, y="outcome", backend="bokeh", show=False, **kwargs)
    assert ax


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_names": "mu"},
        {"var_names": ["mu", "tau"]},
        {"combined": True, "rug": True},
        {"compact": True, "legend": True},
        {"combined": True, "compact": True, "legend": True},
        {"divergences": "top"},
        {"divergences": False},
        {"kind": "rank_vlines"},
        {"kind": "rank_bars"},
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
    with pytest.warns(UserWarning):
        with rc_context(rc={"plot.max_subplots": 2}):
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
    axes = plot_dist(continuous_model["x"], backend="bokeh", show=False, **kwargs)
    assert axes


def test_plot_kde_1d(continuous_model):
    axes = plot_kde(continuous_model["y"], backend="bokeh", show=False)
    assert axes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contour": True, "fill_last": False},
        {"contour": True, "contourf_kwargs": {"cmap": "plasma"}},
        {"contour": False},
        {"contour": False, "pcolormesh_kwargs": {"cmap": "plasma"}},
        {"contour": True, "contourf_kwargs": {"levels": 3}},
        {"contour": True, "contourf_kwargs": {"levels": [0.1, 0.2, 0.3]}},
        {"hdi_probs": [0.3, 0.9, 0.6]},
        {"hdi_probs": [0.3, 0.6, 0.9], "contourf_kwargs": {"cmap": "Blues"}},
        {"hdi_probs": [0.9, 0.6, 0.3], "contour_kwargs": {"alpha": 0}},
    ],
)
def test_plot_kde_2d(continuous_model, kwargs):
    axes = plot_kde(
        continuous_model["x"], continuous_model["y"], backend="bokeh", show=False, **kwargs
    )
    assert axes


@pytest.mark.parametrize(
    "kwargs", [{"plot_kwargs": {"line_dash": "solid"}}, {"cumulative": True}, {"rug": True}]
)
def test_plot_kde_quantiles(continuous_model, kwargs):
    axes = plot_kde(
        continuous_model["x"], quantiles=[0.05, 0.5, 0.95], backend="bokeh", show=False, **kwargs
    )
    assert axes


def test_plot_autocorr_short_chain():
    """Check that logic for small chain defaulting doesn't cause exception"""
    chain = np.arange(10)
    axes = plot_autocorr(chain, backend="bokeh", show=False)
    assert axes


def test_plot_autocorr_uncombined(models):
    axes = plot_autocorr(models.model_1, combined=False, backend="bokeh", show=False)
    assert axes.shape[0] == 10
    max_subplots = (
        np.inf if rcParams["plot.max_subplots"] is None else rcParams["plot.max_subplots"]
    )
    assert len([ax for ax in axes.ravel() if ax is not None]) == min(72, max_subplots)


def test_plot_autocorr_combined(models):
    axes = plot_autocorr(models.model_1, combined=True, backend="bokeh", show=False)
    assert axes.shape[0] == 6
    assert axes.shape[1] == 3


@pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
def test_plot_autocorr_var_names(models, var_names):
    axes = plot_autocorr(
        models.model_1, var_names=var_names, combined=True, backend="bokeh", show=False
    )
    assert axes.shape


@pytest.mark.parametrize(
    "kwargs", [{"insample_dev": False}, {"plot_standard_error": False}, {"plot_ic_diff": False}]
)
def test_plot_compare(models, kwargs):

    model_compare = compare({"Model 1": models.model_1, "Model 2": models.model_2})

    axes = plot_compare(model_compare, backend="bokeh", show=False, **kwargs)
    assert axes


def test_plot_compare_no_ic(models):
    """Check exception is raised if model_compare doesn't contain a valid information criterion"""
    model_compare = compare({"Model 1": models.model_1, "Model 2": models.model_2})

    # Drop column needed for plotting
    model_compare = model_compare.drop("loo", axis=1)
    with pytest.raises(ValueError) as err:
        plot_compare(model_compare, backend="bokeh", show=False)

    assert "comp_df must contain one of the following" in str(err.value)
    assert "['loo', 'waic']" in str(err.value)


def test_plot_ecdf_basic():
    data = np.random.randn(4, 1000)
    axes = plot_ecdf(data, backend="bokeh", show=False)
    assert axes is not None


def test_plot_ecdf_values2():
    data = np.random.randn(4, 1000)
    data2 = np.random.randn(4, 500)
    axes = plot_ecdf(data, data2, backend="bokeh", show=False)
    assert axes is not None


def test_plot_ecdf_cdf():
    data = np.random.randn(4, 1000)
    cdf = norm(0, 1).cdf
    axes = plot_ecdf(data, cdf=cdf, backend="bokeh", show=False)
    assert axes is not None


@pytest.mark.parametrize(
    "kwargs", [{}, {"ic": "loo"}, {"xlabels": True, "scale": "log"}, {"threshold": 2}]
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

    axes = plot_elpd(model_dict, backend="bokeh", show=False, **kwargs)
    assert np.any(axes)
    if add_model:
        assert axes.shape[0] == axes.shape[1]
        assert axes.shape[0] == len(model_dict) - 1


@pytest.mark.parametrize("kwargs", [{}, {"ic": "loo"}, {"xlabels": True, "scale": "log"}])
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

    axes = plot_elpd(model_dict, backend="bokeh", show=False, **kwargs)
    assert np.any(axes)
    if add_model:
        assert axes.shape[0] == axes.shape[1]
        assert axes.shape[0] == len(model_dict) - 1


@pytest.mark.parametrize("kind", ["kde", "hist"])
def test_plot_energy(models, kind):
    assert plot_energy(models.model_1, kind=kind, backend="bokeh", show=False)


def test_plot_energy_bad(models):
    with pytest.raises(ValueError):
        plot_energy(models.model_1, kind="bad_kind", backend="bokeh", show=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_names": ["theta"], "relative": True, "color": "r"},
        {"coords": {"school": slice(4)}, "n_points": 10},
        {"min_ess": 600, "hline_kwargs": {"line_color": "red"}},
    ],
)
@pytest.mark.parametrize("kind", ["local", "quantile", "evolution"])
def test_plot_ess(models, kind, kwargs):
    """Test plot_ess arguments common to all kind of plots."""
    idata = models.model_1
    ax = plot_ess(idata, kind=kind, backend="bokeh", show=False, **kwargs)
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
    ax = plot_ess(idata, kind=kind, backend="bokeh", show=False, **kwargs)
    assert np.all(ax)


def test_plot_ess_evolution(models):
    """Test specific arguments in evolution kind of plot_ess."""
    idata = models.model_1
    ax = plot_ess(
        idata,
        kind="evolution",
        extra_kwargs={"linestyle": "--"},
        color="b",
        backend="bokeh",
        show=False,
    )
    assert np.all(ax)


def test_plot_ess_bad_kind(models):
    """Test error when plot_ess receives an invalid kind."""
    idata = models.model_1
    with pytest.raises(ValueError, match="Invalid kind"):
        plot_ess(idata, kind="bad kind", backend="bokeh", show=False)


@pytest.mark.parametrize("dim", ["chain", "draw"])
def test_plot_ess_bad_coords(models, dim):
    """Test error when chain or dim are used as coords to select a data subset."""
    idata = models.model_1
    with pytest.raises(ValueError, match="invalid coordinates"):
        plot_ess(idata, coords={dim: slice(3)}, backend="bokeh", show=False)


def test_plot_ess_no_sample_stats(models):
    """Test error when rug=True but sample_stats group is not present."""
    idata = models.model_1
    with pytest.raises(ValueError, match="must contain sample_stats"):
        plot_ess(idata.posterior, rug=True, backend="bokeh", show=False)


def test_plot_ess_no_divergences(models):
    """Test error when rug=True, but the variable defined by rug_kind is missing."""
    idata = deepcopy(models.model_1)
    idata.sample_stats = idata.sample_stats.rename({"diverging": "diverging_missing"})
    with pytest.raises(ValueError, match="not contain diverging"):
        plot_ess(idata, rug=True, backend="bokeh", show=False)


@pytest.mark.parametrize("model_fits", [["model_1"], ["model_1", "model_2"]])
@pytest.mark.parametrize(
    "args_expected",
    [
        ({}, 1),
        ({"var_names": "mu"}, 1),
        ({"var_names": "mu", "rope": (-1, 1)}, 1),
        ({"r_hat": True, "quartiles": False}, 2),
        ({"var_names": ["mu"], "colors": "black", "ess": True, "combined": True}, 2),
        (
            {
                "kind": "ridgeplot",
                "ridgeplot_truncate": False,
                "ridgeplot_quantiles": [0.25, 0.5, 0.75],
            },
            1,
        ),
        ({"kind": "ridgeplot", "r_hat": True, "ess": True}, 3),
        ({"kind": "ridgeplot", "r_hat": True, "ess": True, "ridgeplot_alpha": 0}, 3),
        (
            {
                "var_names": ["mu", "tau"],
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
    axes = plot_forest(obj, backend="bokeh", show=False, **args)
    assert axes.shape == (1, expected)


def test_plot_forest_rope_exception():
    with pytest.raises(ValueError) as err:
        plot_forest({"x": [1]}, rope="not_correct_format", backend="bokeh", show=False)
    assert "Argument `rope` must be None, a dictionary like" in str(err.value)


def test_plot_forest_single_value():
    axes = plot_forest({"x": [1]}, backend="bokeh", show=False)
    assert axes.shape


@pytest.mark.parametrize("model_fits", [["model_1"], ["model_1", "model_2"]])
def test_plot_forest_bad(models, model_fits):
    obj = [getattr(models, model_fit) for model_fit in model_fits]
    with pytest.raises(TypeError):
        plot_forest(obj, kind="bad_kind", backend="bokeh", show=False)

    with pytest.raises(ValueError):
        plot_forest(
            obj,
            model_names=[f"model_name_{i}" for i in range(len(obj) + 10)],
            backend="bokeh",
            show=False,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"color": "C5", "circular": True},
        {"hdi_data": True, "fill_kwargs": {"alpha": 0}},
        {"plot_kwargs": {"alpha": 0}},
        {"smooth_kwargs": {"window_length": 33, "polyorder": 5, "mode": "mirror"}},
        {"hdi_data": True, "smooth": False, "color": "xkcd:jade"},
    ],
)
def test_plot_hdi(models, data, kwargs):
    hdi_data = kwargs.pop("hdi_data", None)
    y_data = models.model_1.posterior["theta"]
    if hdi_data:
        hdi_data = hdi(y_data)
        axis = plot_hdi(data["y"], hdi_data=hdi_data, backend="bokeh", show=False, **kwargs)
    else:
        axis = plot_hdi(data["y"], y_data, backend="bokeh", show=False, **kwargs)
    assert axis


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

    axes = plot_khat(khats_data, backend="bokeh", show=False, **kwargs)
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

    axes = plot_khat(khats_data, backend="bokeh", show=False, **kwargs)
    assert axes


def test_plot_khat_threshold():
    khats = np.array([0, 0, 0.6, 0.6, 0.8, 0.9, 0.9, 2, 3, 4, 1.5])
    axes = plot_khat(khats, threshold=1, backend="bokeh", show=False)
    assert axes


def test_plot_khat_bad_input(models):
    with pytest.raises(ValueError):
        plot_khat(models.model_1.sample_stats, backend="bokeh", show=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_unif": 50},
        {"use_hdi": True, "color": "gray"},
        {"use_hdi": True, "hdi_prob": 0.68},
        {"use_hdi": True, "hdi_kwargs": {"line_dash": "dashed", "alpha": 0}},
        {"ecdf": True},
        {"ecdf": True, "ecdf_fill": False, "plot_unif_kwargs": {"line_dash": "--"}},
        {"ecdf": True, "hdi_prob": 0.97, "fill_kwargs": {"color": "red"}},
    ],
)
def test_plot_loo_pit(models, kwargs):
    axes = plot_loo_pit(idata=models.model_1, y="y", backend="bokeh", show=False, **kwargs)
    assert axes


def test_plot_loo_pit_incompatible_args(models):
    """Test error when both ecdf and use_hdi are True."""
    with pytest.raises(ValueError, match="incompatible"):
        plot_loo_pit(
            idata=models.model_1, y="y", ecdf=True, use_hdi=True, backend="bokeh", show=False
        )


@pytest.mark.parametrize(
    "args",
    [
        {"y": "str"},
        {"y": "DataArray", "y_hat": "str"},
        {"y": "ndarray", "y_hat": "str"},
        {"y": "ndarray", "y_hat": "DataArray"},
        {"y": "ndarray", "y_hat": "ndarray"},
    ],
)
def test_plot_loo_pit_label(models, args):
    if args["y"] == "str":
        y = "y"
    elif args["y"] == "DataArray":
        y = models.model_1.observed_data.y
    elif args["y"] == "ndarray":
        y = models.model_1.observed_data.y.values

    if args.get("y_hat") == "str":
        y_hat = "y"
    elif args.get("y_hat") == "DataArray":
        y_hat = models.model_1.posterior_predictive.y.stack(__sample__=("chain", "draw"))
    elif args.get("y_hat") == "ndarray":
        y_hat = models.model_1.posterior_predictive.y.stack(__sample__=("chain", "draw")).values
    else:
        y_hat = None

    ax = plot_loo_pit(idata=models.model_1, y=y, y_hat=y_hat, backend="bokeh", show=False)
    assert ax


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
    ax = plot_mcse(idata, backend="bokeh", show=False, **kwargs)
    assert np.all(ax)


@pytest.mark.parametrize("dim", ["chain", "draw"])
def test_plot_mcse_bad_coords(models, dim):
    """Test error when chain or dim are used as coords to select a data subset."""
    idata = models.model_1
    with pytest.raises(ValueError, match="invalid coordinates"):
        plot_mcse(idata, coords={dim: slice(3)}, backend="bokeh", show=False)


def test_plot_mcse_no_sample_stats(models):
    """Test error when rug=True but sample_stats group is not present."""
    idata = models.model_1
    with pytest.raises(ValueError, match="must contain sample_stats"):
        plot_mcse(idata.posterior, rug=True, backend="bokeh", show=False)


def test_plot_mcse_no_divergences(models):
    """Test error when rug=True, but the variable defined by rug_kind is missing."""
    idata = deepcopy(models.model_1)
    idata.sample_stats = idata.sample_stats.rename({"diverging": "diverging_missing"})
    with pytest.raises(ValueError, match="not contain diverging"):
        plot_mcse(idata, rug=True, backend="bokeh", show=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "kwargs",
    [
        {"var_names": "theta", "divergences": True, "coords": {"school": [0, 1]}},
        {"divergences": True, "var_names": ["theta", "mu"]},
        {"kind": "kde", "var_names": ["theta"]},
        {"kind": "hexbin", "var_names": ["theta"]},
        {"kind": "hexbin", "var_names": ["theta"]},
        {
            "kind": "hexbin",
            "var_names": ["theta"],
            "coords": {"school": [0, 1]},
            "textsize": 20,
        },
        {
            "point_estimate": "mean",
            "reference_values": {"mu": 0, "tau": 0},
            "reference_values_kwargs": {"line_color": "blue"},
        },
    ],
)
def test_plot_pair(models, kwargs):
    ax = plot_pair(models.model_1, backend="bokeh", show=False, **kwargs)
    assert np.any(ax)


@pytest.mark.parametrize("kwargs", [{"kind": "scatter"}, {"kind": "kde"}, {"kind": "hexbin"}])
def test_plot_pair_2var(discrete_model, kwargs):
    ax = plot_pair(
        discrete_model, ax=np.atleast_2d(bkp.figure()), backend="bokeh", show=False, **kwargs
    )
    assert ax


def test_plot_pair_bad(models):
    with pytest.raises(ValueError):
        plot_pair(models.model_1, kind="bad_kind", backend="bokeh", show=False)
    with pytest.raises(Exception):
        plot_pair(models.model_1, var_names=["mu"], backend="bokeh", show=False)


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
        ax = plot_pair(data, divergences=True, backend="bokeh", show=False)
    assert np.any(ax)


def test_plot_parallel_raises_valueerror(df_trace):  # pylint: disable=invalid-name
    with pytest.raises(ValueError):
        plot_parallel(df_trace, backend="bokeh", show=False)


@pytest.mark.parametrize("norm_method", [None, "normal", "minmax", "rank"])
def test_plot_parallel(models, norm_method):
    assert plot_parallel(
        models.model_1,
        var_names=["mu", "tau"],
        norm_method=norm_method,
        backend="bokeh",
        show=False,
    )


@pytest.mark.parametrize("var_names", [None, "mu", ["mu", "tau"]])
def test_plot_parallel_exception(models, var_names):
    """Ensure that correct exception is raised when one variable is passed."""
    with pytest.raises(ValueError):
        assert plot_parallel(
            models.model_1, var_names=var_names, norm_method="foo", backend="bokeh", show=False
        )


@pytest.mark.parametrize("var_names", (None, "mu", ["mu", "tau"]))
def test_plot_violin(models, var_names):
    axes = plot_violin(models.model_1, var_names=var_names, backend="bokeh", show=False)
    assert axes.shape


def test_plot_violin_ax(models):
    ax = bkp.figure()
    axes = plot_violin(models.model_1, var_names="mu", ax=ax, backend="bokeh", show=False)
    assert axes.shape


def test_plot_violin_layout(models):
    axes = plot_violin(
        models.model_1, var_names=["mu", "tau"], sharey=False, backend="bokeh", show=False
    )
    assert axes.shape


def test_plot_violin_discrete(discrete_model):
    axes = plot_violin(discrete_model, backend="bokeh", show=False)
    assert axes.shape


@pytest.mark.parametrize("kind", ["kde", "cumulative", "scatter"])
@pytest.mark.parametrize("alpha", [None, 0.2, 1])
@pytest.mark.parametrize("observed", [True, False])
def test_plot_ppc(models, kind, alpha, observed):
    axes = plot_ppc(
        models.model_1,
        kind=kind,
        alpha=alpha,
        observed=observed,
        random_seed=3,
        backend="bokeh",
        show=False,
    )
    assert axes


def test_plot_ppc_textsize(models):
    axes = plot_ppc(
        models.model_1,
        textsize=10,
        random_seed=3,
        backend="bokeh",
        show=False,
    )
    assert axes


@pytest.mark.parametrize("kind", ["kde", "cumulative", "scatter"])
@pytest.mark.parametrize("jitter", [None, 0, 0.1, 1, 3])
def test_plot_ppc_multichain(kind, jitter):
    data = from_dict(
        posterior_predictive={
            "x": np.random.randn(4, 100, 30),
            "y_hat": np.random.randn(4, 100, 3, 10),
        },
        observed_data={"x": np.random.randn(30), "y": np.random.randn(3, 10)},
    )
    axes = plot_ppc(
        data,
        kind=kind,
        data_pairs={"y": "y_hat"},
        jitter=jitter,
        random_seed=3,
        backend="bokeh",
        show=False,
    )
    assert np.all(axes)


@pytest.mark.parametrize("kind", ["kde", "cumulative", "scatter"])
def test_plot_ppc_discrete(kind):
    data = from_dict(
        observed_data={"obs": np.random.randint(1, 100, 15)},
        posterior_predictive={"obs": np.random.randint(1, 300, (1, 20, 15))},
    )

    axes = plot_ppc(data, kind=kind, backend="bokeh", show=False)
    assert axes


def test_plot_ppc_grid(models):
    axes = plot_ppc(models.model_1, kind="scatter", flatten=[], backend="bokeh", show=False)
    assert len(axes.ravel()) == 8
    axes = plot_ppc(
        models.model_1,
        kind="scatter",
        flatten=[],
        coords={"obs_dim": [1, 2, 3]},
        backend="bokeh",
        show=False,
    )
    assert len(axes.ravel()) == 3
    axes = plot_ppc(
        models.model_1,
        kind="scatter",
        flatten=["obs_dim"],
        coords={"obs_dim": [1, 2, 3]},
        backend="bokeh",
        show=False,
    )
    assert len(axes.ravel()) == 1


@pytest.mark.parametrize("kind", ["kde", "cumulative", "scatter"])
def test_plot_ppc_bad(models, kind):
    data = from_dict(posterior={"mu": np.random.randn()})
    with pytest.raises(TypeError):
        plot_ppc(data, kind=kind, backend="bokeh", show=False)
    with pytest.raises(TypeError):
        plot_ppc(models.model_1, kind="bad_val", backend="bokeh", show=False)
    with pytest.raises(TypeError):
        plot_ppc(models.model_1, num_pp_samples="bad_val", backend="bokeh", show=False)


@pytest.mark.parametrize("kind", ["kde", "cumulative", "scatter"])
def test_plot_ppc_ax(models, kind):
    """Test ax argument of plot_ppc."""
    ax = bkp.figure()
    axes = plot_ppc(models.model_1, kind=kind, ax=ax, backend="bokeh", show=False)
    assert axes[0, 0] is ax


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
        {"point_estimate": None},
        {"hdi_prob": "hide", "legend_label": ""},
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
    axes = plot_posterior(models.model_1, backend="bokeh", show=False, **kwargs)
    assert axes.shape


@pytest.mark.parametrize("kwargs", [{}, {"point_estimate": "mode"}, {"bins": None, "kind": "hist"}])
def test_plot_posterior_discrete(discrete_model, kwargs):
    axes = plot_posterior(discrete_model, backend="bokeh", show=False, **kwargs)
    assert axes.shape


def test_plot_posterior_boolean():
    data = np.random.choice(a=[False, True], size=(4, 100))
    axes = plot_posterior(data, backend="bokeh", show=False)
    assert axes.shape


def test_plot_posterior_bad_type():
    with pytest.raises(TypeError):
        plot_posterior(np.array(["a", "b", "c"]), backend="bokeh", show=False)


def test_plot_posterior_bad(models):
    with pytest.raises(ValueError):
        plot_posterior(models.model_1, backend="bokeh", show=False, rope="bad_value")
    with pytest.raises(ValueError):
        plot_posterior(models.model_1, ref_val="bad_value", backend="bokeh", show=False)
    with pytest.raises(ValueError):
        plot_posterior(models.model_1, point_estimate="bad_value", backend="bokeh", show=False)


@pytest.mark.parametrize("point_estimate", ("mode", "mean", "median"))
def test_plot_posterior_point_estimates(models, point_estimate):
    axes = plot_posterior(
        models.model_1,
        var_names=("mu", "tau"),
        point_estimate=point_estimate,
        backend="bokeh",
        show=False,
    )
    assert axes.shape == (1, 2)


def test_plot_posterior_skipna():
    sample = np.linspace(0, 1)
    sample[:10] = np.nan
    plot_posterior({"a": sample}, backend="bokeh", show=False, skipna=True)
    with pytest.raises(ValueError):
        plot_posterior({"a": sample}, backend="bokeh", show=False, skipna=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"var_names": "mu"},
        {"var_names": ("mu", "tau"), "coords": {"school": [0, 1]}},
        {"var_names": "mu", "ref_line": True},
        {
            "var_names": "mu",
            "ref_line_kwargs": {"line_width": 2, "line_color": "red"},
            "bar_kwargs": {"width": 50},
        },
        {"var_names": "mu", "ref_line": False},
        {"var_names": "mu", "kind": "vlines"},
        {
            "var_names": "mu",
            "kind": "vlines",
            "vlines_kwargs": {"line_width": 0},
            "marker_vlines_kwargs": {"radius": 20},
        },
    ],
)
def test_plot_rank(models, kwargs):
    axes = plot_rank(models.model_1, backend="bokeh", show=False, **kwargs)
    assert axes.shape


def test_plot_dist_comparison_warn(models):
    with pytest.raises(NotImplementedError, match="The bokeh backend.+Use matplotlib backend."):
        plot_dist_comparison(models.model_1, backend="bokeh")


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
    axes = plot_bpv(models.model_1, backend="bokeh", show=False, **kwargs)
    assert axes.shape


def test_plot_bpv_discrete():
    fake_obs = {"a": np.random.poisson(2.5, 100)}
    fake_pp = {"a": np.random.poisson(2.5, (1, 10, 100))}
    fake_model = from_dict(posterior_predictive=fake_pp, observed_data=fake_obs)
    axes = plot_bpv(
        fake_model,
        backend="bokeh",
        show=False,
    )
    assert axes.shape


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
    ax = plot_dot(data, **kwargs, backend="bokeh", show=False)
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
    ax = plot_dot(data, **kwargs, backend="bokeh", show=False)
    assert ax


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"y_hat": "bad_name"},
        {"x": "x1"},
        {"x": ("x1", "x2")},
        {
            "x": ("x1", "x2"),
            "y_kwargs": {"fill_color": "blue"},
            "y_hat_plot_kwargs": {"fill_color": "orange"},
            "legend": True,
        },
        {"x": ("x1", "x2"), "y_model_plot_kwargs": {"line_color": "red"}},
        {
            "x": ("x1", "x2"),
            "kind_pp": "hdi",
            "kind_model": "hdi",
            "y_model_fill_kwargs": {"color": "red"},
            "y_hat_fill_kwargs": {"color": "cyan"},
        },
    ],
)
def test_plot_lm(models, kwargs):
    """Test functionality for 1D data."""
    idata = models.model_1
    if "constant_data" not in idata.groups():
        y = idata.observed_data["y"]
        x1data = y.coords[y.dims[0]]
        idata.add_groups({"constant_data": {"_": x1data}})
        idata.constant_data["x1"] = x1data
        idata.constant_data["x2"] = x1data

    axes = plot_lm(
        idata=idata, y="y", y_model="eta", backend="bokeh", xjitter=True, show=False, **kwargs
    )
    assert np.all(axes)


def test_plot_lm_multidim(multidim_models):
    """Test functionality for multidimentional data."""
    idata = multidim_models.model_1
    axes = plot_lm(idata=idata, y="y", plot_dim="dim1", show=False, backend="bokeh")
    assert np.any(axes)


def test_plot_lm_list():
    """Test the plots when input data is list or ndarray."""
    y = [1, 2, 3, 4, 5]
    assert plot_lm(y=y, x=np.arange(len(y)), show=False, backend="bokeh")
