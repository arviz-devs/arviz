# pylint: disable=redefined-outer-name, no-member
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from scipy.special import logsumexp
from scipy.stats import linregress
from xarray import DataArray, Dataset

from ...data import concat, convert_to_inference_data, from_dict, load_arviz_data
from ...rcparams import rcParams
from ...stats import (
    apply_test_function,
    compare,
    ess,
    hdi,
    loo,
    loo_pit,
    psislw,
    r2_score,
    summary,
    waic,
    _calculate_ics,
)
from ...stats.stats import _gpinv
from ...stats.stats_utils import get_log_likelihood
from ..helpers import check_multiple_attrs, multidim_models  # pylint: disable=unused-import

rcParams["data.load"] = "eager"


@pytest.fixture(scope="session")
def centered_eight():
    centered_eight = load_arviz_data("centered_eight")
    return centered_eight


@pytest.fixture(scope="session")
def non_centered_eight():
    non_centered_eight = load_arviz_data("non_centered_eight")
    return non_centered_eight


@pytest.fixture(scope="module")
def multivariable_log_likelihood(centered_eight):
    centered_eight = centered_eight.copy()
    centered_eight.add_groups({"log_likelihood": centered_eight.sample_stats.log_likelihood})
    centered_eight.log_likelihood = centered_eight.log_likelihood.rename_vars(
        {"log_likelihood": "obs"}
    )
    new_arr = DataArray(
        np.zeros(centered_eight.log_likelihood["obs"].values.shape),
        dims=["chain", "draw", "school"],
        coords=centered_eight.log_likelihood.coords,
    )
    centered_eight.log_likelihood["decoy"] = new_arr
    delattr(centered_eight, "sample_stats")
    return centered_eight


def test_hdp():
    normal_sample = np.random.randn(5000000)
    interval = hdi(normal_sample)
    assert_array_almost_equal(interval, [-1.88, 1.88], 2)


def test_hdp_2darray():
    normal_sample = np.random.randn(12000, 5)
    msg = (
        r"hdi currently interprets 2d data as \(draw, shape\) but this will "
        r"change in a future release to \(chain, draw\) for coherence with other functions"
    )
    with pytest.warns(FutureWarning, match=msg):
        result = hdi(normal_sample)
    assert result.shape == (5, 2)


def test_hdi_multidimension():
    normal_sample = np.random.randn(12000, 10, 3)
    result = hdi(normal_sample)
    assert result.shape == (3, 2)


def test_hdi_idata(centered_eight):
    data = centered_eight.posterior
    result = hdi(data)
    assert isinstance(result, Dataset)
    assert dict(result.dims) == {"school": 8, "hdi": 2}

    result = hdi(data, input_core_dims=[["chain"]])
    assert isinstance(result, Dataset)
    assert result.dims == {"draw": 500, "hdi": 2, "school": 8}


def test_hdi_idata_varnames(centered_eight):
    data = centered_eight.posterior
    result = hdi(data, var_names=["mu", "theta"])
    assert isinstance(result, Dataset)
    assert result.dims == {"hdi": 2, "school": 8}
    assert list(result.data_vars.keys()) == ["mu", "theta"]


def test_hdi_idata_group(centered_eight):
    result_posterior = hdi(centered_eight, group="posterior", var_names="mu")
    result_prior = hdi(centered_eight, group="prior", var_names="mu")
    assert result_prior.dims == {"hdi": 2}
    range_posterior = result_posterior.mu.values[1] - result_posterior.mu.values[0]
    range_prior = result_prior.mu.values[1] - result_prior.mu.values[0]
    assert range_posterior < range_prior


def test_hdi_coords(centered_eight):
    data = centered_eight.posterior
    result = hdi(data, coords={"chain": [0, 1, 3]}, input_core_dims=[["draw"]])
    assert_array_equal(result.coords["chain"], [0, 1, 3])


def test_hdi_multimodal():
    normal_sample = np.concatenate(
        (np.random.normal(-4, 1, 2500000), np.random.normal(2, 0.5, 2500000))
    )
    intervals = hdi(normal_sample, multimodal=True)
    assert_array_almost_equal(intervals, [[-5.8, -2.2], [0.9, 3.1]], 1)


def test_hdi_multimodal_multivars():
    size = 2500000
    var1 = np.concatenate((np.random.normal(-4, 1, size), np.random.normal(2, 0.5, size)))
    var2 = np.random.normal(8, 1, size * 2)
    sample = Dataset(
        {
            "var1": (("chain", "draw"), var1[np.newaxis, :]),
            "var2": (("chain", "draw"), var2[np.newaxis, :]),
        },
        coords={"chain": [0], "draw": np.arange(size * 2)},
    )
    intervals = hdi(sample, multimodal=True)
    assert_array_almost_equal(intervals.var1, [[-5.8, -2.2], [0.9, 3.1]], 1)
    assert_array_almost_equal(intervals.var2, [[6.1, 9.9], [np.nan, np.nan]], 1)


def test_hdi_circular():
    normal_sample = np.random.vonmises(np.pi, 1, 5000000)
    interval = hdi(normal_sample, circular=True)
    assert_array_almost_equal(interval, [0.6, -0.6], 1)


def test_hdi_bad_ci():
    normal_sample = np.random.randn(10)
    with pytest.raises(ValueError):
        hdi(normal_sample, hdi_prob=2)


def test_hdi_skipna():
    normal_sample = np.random.randn(500)
    interval = hdi(normal_sample[10:])
    normal_sample[:10] = np.nan
    interval_ = hdi(normal_sample, skipna=True)
    assert_array_almost_equal(interval, interval_)


def test_r2_score():
    x = np.linspace(0, 1, 100)
    y = np.random.normal(x, 1)
    y_pred = x + np.random.randn(300, 100)
    res = linregress(x, y)
    assert_allclose(res.rvalue**2, r2_score(y, y_pred).r2, 2)


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
@pytest.mark.parametrize("multidim", [True, False])
def test_compare_same(centered_eight, multidim_models, method, multidim):
    if multidim:
        data_dict = {"first": multidim_models.model_1, "second": multidim_models.model_1}
    else:
        data_dict = {"first": centered_eight, "second": centered_eight}

    weight = compare(data_dict, method=method)["weight"]
    assert_allclose(weight[0], weight[1])
    assert_allclose(np.sum(weight), 1.0)


def test_compare_unknown_ic_and_method(centered_eight, non_centered_eight):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    with pytest.raises(ValueError):
        compare(model_dict, ic="Unknown", method="stacking")
    with pytest.raises(ValueError):
        compare(model_dict, ic="loo", method="Unknown")


@pytest.mark.parametrize("ic", ["loo", "waic"])
@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_compare_different(centered_eight, non_centered_eight, ic, method, scale):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    weight = compare(model_dict, ic=ic, method=method, scale=scale)["weight"]
    assert weight["non_centered"] > weight["centered"]
    assert_allclose(np.sum(weight), 1.0)


@pytest.mark.parametrize("ic", ["loo", "waic"])
@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_different_multidim(multidim_models, ic, method):
    model_dict = {"model_1": multidim_models.model_1, "model_2": multidim_models.model_2}
    weight = compare(model_dict, ic=ic, method=method)["weight"]

    # this should hold because the same seed is always used
    assert weight["model_1"] > weight["model_2"]
    assert_allclose(np.sum(weight), 1.0)


def test_compare_different_size(centered_eight, non_centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior = centered_eight.posterior.drop("Choate", "school")
    centered_eight.sample_stats = centered_eight.sample_stats.drop("Choate", "school")
    centered_eight.posterior_predictive = centered_eight.posterior_predictive.drop(
        "Choate", "school"
    )
    centered_eight.prior = centered_eight.prior.drop("Choate", "school")
    centered_eight.observed_data = centered_eight.observed_data.drop("Choate", "school")
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    with pytest.raises(ValueError):
        compare(model_dict, ic="waic", method="stacking")


@pytest.mark.parametrize("ic", ["loo", "waic"])
def test_compare_multiple_obs(multivariable_log_likelihood, centered_eight, non_centered_eight, ic):
    compare_dict = {
        "centered_eight": centered_eight,
        "non_centered_eight": non_centered_eight,
        "problematic": multivariable_log_likelihood,
    }
    with pytest.raises(TypeError, match="several log likelihood arrays"):
        get_log_likelihood(compare_dict["problematic"])
    with pytest.raises(TypeError, match="error in ic computation"):
        compare(compare_dict, ic=ic)
    assert compare(compare_dict, ic=ic, var_name="obs") is not None


@pytest.mark.parametrize("ic", ["loo", "waic"])
def test_calculate_ics(centered_eight, non_centered_eight, ic):
    ic_func = loo if ic == "loo" else waic
    idata_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    elpddata_dict = {key: ic_func(value) for key, value in idata_dict.items()}
    mixed_dict = {"centered": idata_dict["centered"], "non_centered": elpddata_dict["non_centered"]}
    idata_out, _, _ = _calculate_ics(idata_dict, ic=ic)
    elpddata_out, _, _ = _calculate_ics(elpddata_dict, ic=ic)
    mixed_out, _, _ = _calculate_ics(mixed_dict, ic=ic)
    for model in idata_dict:
        assert idata_out[model][ic] == elpddata_out[model][ic]
        assert idata_out[model][ic] == mixed_out[model][ic]
        assert idata_out[model][f"p_{ic}"] == elpddata_out[model][f"p_{ic}"]
        assert idata_out[model][f"p_{ic}"] == mixed_out[model][f"p_{ic}"]


def test_calculate_ics_ic_error(centered_eight, non_centered_eight):
    in_dict = {"centered": loo(centered_eight), "non_centered": waic(non_centered_eight)}
    with pytest.raises(ValueError, match="found both loo and waic"):
        _calculate_ics(in_dict)


def test_calculate_ics_ic_override(centered_eight, non_centered_eight):
    in_dict = {"centered": centered_eight, "non_centered": waic(non_centered_eight)}
    with pytest.warns(UserWarning, match="precomputed elpddata: waic"):
        out_dict, _, ic = _calculate_ics(in_dict, ic="loo")
    assert ic == "waic"
    assert out_dict["centered"]["waic"] == waic(centered_eight)["waic"]


def test_summary_ndarray():
    array = np.random.randn(4, 100, 2)
    summary_df = summary(array)
    assert summary_df.shape


@pytest.mark.parametrize("var_names_expected", ((None, 10), ("mu", 1), (["mu", "tau"], 2)))
def test_summary_var_names(centered_eight, var_names_expected):
    var_names, expected = var_names_expected
    summary_df = summary(centered_eight, var_names=var_names)
    assert len(summary_df.index) == expected


@pytest.mark.parametrize("missing_groups", (None, "posterior", "prior"))
def test_summary_groups(centered_eight, missing_groups):
    if missing_groups == "posterior":
        centered_eight = deepcopy(centered_eight)
        del centered_eight.posterior
    elif missing_groups == "prior":
        centered_eight = deepcopy(centered_eight)
        del centered_eight.posterior
        del centered_eight.prior
    if missing_groups == "prior":
        with pytest.warns(UserWarning):
            summary_df = summary(centered_eight)
    else:
        summary_df = summary(centered_eight)
    assert summary_df.shape


def test_summary_group_argument(centered_eight):
    summary_df_posterior = summary(centered_eight, group="posterior")
    summary_df_prior = summary(centered_eight, group="prior")
    assert list(summary_df_posterior.index) != list(summary_df_prior.index)


def test_summary_wrong_group(centered_eight):
    with pytest.raises(TypeError, match=r"InferenceData does not contain group: InvalidGroup"):
        summary(centered_eight, group="InvalidGroup")


METRICS_NAMES = [
    "mean",
    "sd",
    "hdi_3%",
    "hdi_97%",
    "mcse_mean",
    "mcse_sd",
    "ess_bulk",
    "ess_tail",
    "r_hat",
    "median",
    "mad",
    "eti_3%",
    "eti_97%",
    "mcse_median",
    "ess_median",
    "ess_tail",
    "r_hat",
]


@pytest.mark.parametrize(
    "params",
    (
        ("mean", "all", METRICS_NAMES[:9]),
        ("mean", "stats", METRICS_NAMES[:4]),
        ("mean", "diagnostics", METRICS_NAMES[4:9]),
        ("median", "all", METRICS_NAMES[9:17]),
        ("median", "stats", METRICS_NAMES[9:13]),
        ("median", "diagnostics", METRICS_NAMES[13:17]),
    ),
)
def test_summary_focus_kind(centered_eight, params):
    stat_focus, kind, metrics_names_ = params
    summary_df = summary(centered_eight, stat_focus=stat_focus, kind=kind)
    assert_array_equal(summary_df.columns, metrics_names_)


def test_summary_wrong_focus(centered_eight):
    with pytest.raises(TypeError, match=r"Invalid format: 'WrongFocus'.*"):
        summary(centered_eight, stat_focus="WrongFocus")


@pytest.mark.parametrize("fmt", ["wide", "long", "xarray"])
def test_summary_fmt(centered_eight, fmt):
    assert summary(centered_eight, fmt=fmt) is not None


def test_summary_labels():
    coords1 = list("abcd")
    coords2 = np.arange(1, 6)
    data = from_dict(
        {"a": np.random.randn(4, 100, 4, 5)},
        coords={"dim1": coords1, "dim2": coords2},
        dims={"a": ["dim1", "dim2"]},
    )
    az_summary = summary(data, fmt="wide")
    assert az_summary is not None
    column_order = []
    for coord1 in coords1:
        for coord2 in coords2:
            column_order.append(f"a[{coord1}, {coord2}]")
    for col1, col2 in zip(list(az_summary.index), column_order):
        assert col1 == col2


@pytest.mark.parametrize(
    "stat_funcs", [[np.var], {"var": np.var, "var2": lambda x: np.var(x) ** 2}]
)
def test_summary_stat_func(centered_eight, stat_funcs):
    arviz_summary = summary(centered_eight, stat_funcs=stat_funcs)
    assert arviz_summary is not None
    assert hasattr(arviz_summary, "var")


def test_summary_nan(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior["theta"].loc[{"school": "Deerfield"}] = np.nan
    summary_xarray = summary(centered_eight)
    assert summary_xarray is not None
    assert summary_xarray.loc["theta[Deerfield]"].isnull().all()
    assert (
        summary_xarray.loc[[ix for ix in summary_xarray.index if ix != "theta[Deerfield]"]]
        .notnull()
        .all()
        .all()
    )


def test_summary_skip_nan(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior["theta"].loc[{"draw": slice(10), "school": "Deerfield"}] = np.nan
    summary_xarray = summary(centered_eight)
    theta_1 = summary_xarray.loc["theta[Deerfield]"].isnull()
    assert summary_xarray is not None
    assert ~theta_1[:4].all()
    assert theta_1[4:].all()


@pytest.mark.parametrize("fmt", [1, "bad_fmt"])
def test_summary_bad_fmt(centered_eight, fmt):
    with pytest.raises(TypeError, match="Invalid format"):
        summary(centered_eight, fmt=fmt)


def test_summary_order_deprecation(centered_eight):
    with pytest.warns(DeprecationWarning, match="order"):
        summary(centered_eight, order="C")


def test_summary_index_origin_deprecation(centered_eight):
    with pytest.warns(DeprecationWarning, match="index_origin"):
        summary(centered_eight, index_origin=1)


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
@pytest.mark.parametrize("multidim", (True, False))
def test_waic(centered_eight, multidim_models, scale, multidim):
    """Test widely available information criterion calculation"""
    if multidim:
        assert waic(multidim_models.model_1, scale=scale) is not None
        waic_pointwise = waic(multidim_models.model_1, pointwise=True, scale=scale)
    else:
        assert waic(centered_eight, scale=scale) is not None
        waic_pointwise = waic(centered_eight, pointwise=True, scale=scale)
    assert waic_pointwise is not None
    assert "waic_i" in waic_pointwise


def test_waic_bad(centered_eight):
    """Test widely available information criterion calculation"""
    centered_eight = deepcopy(centered_eight)
    del centered_eight.sample_stats["log_likelihood"]
    with pytest.raises(TypeError):
        waic(centered_eight)

    del centered_eight.sample_stats
    with pytest.raises(TypeError):
        waic(centered_eight)


def test_waic_bad_scale(centered_eight):
    """Test widely available information criterion calculation with bad scale."""
    with pytest.raises(TypeError):
        waic(centered_eight, scale="bad_value")


def test_waic_warning(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.sample_stats["log_likelihood"][:, :250, 1] = 10
    with pytest.warns(UserWarning):
        assert waic(centered_eight, pointwise=True) is not None
    # this should throw a warning, but due to numerical issues it fails
    centered_eight.sample_stats["log_likelihood"][:, :, :] = 0
    with pytest.warns(UserWarning):
        assert waic(centered_eight, pointwise=True) is not None


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_waic_print(centered_eight, scale):
    waic_data = waic(centered_eight, scale=scale).__repr__()
    waic_pointwise = waic(centered_eight, scale=scale, pointwise=True).__repr__()
    assert waic_data is not None
    assert waic_pointwise is not None
    assert waic_data == waic_pointwise


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
@pytest.mark.parametrize("multidim", (True, False))
def test_loo(centered_eight, multidim_models, scale, multidim):
    """Test approximate leave one out criterion calculation"""
    if multidim:
        assert loo(multidim_models.model_1, scale=scale) is not None
        loo_pointwise = loo(multidim_models.model_1, pointwise=True, scale=scale)
    else:
        assert loo(centered_eight, scale=scale) is not None
        loo_pointwise = loo(centered_eight, pointwise=True, scale=scale)
    assert loo_pointwise is not None
    assert "loo_i" in loo_pointwise
    assert "pareto_k" in loo_pointwise
    assert "loo_scale" in loo_pointwise


def test_loo_one_chain(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior = centered_eight.posterior.drop([1, 2, 3], "chain")
    centered_eight.sample_stats = centered_eight.sample_stats.drop([1, 2, 3], "chain")
    assert loo(centered_eight) is not None


def test_loo_bad(centered_eight):
    with pytest.raises(TypeError):
        loo(np.random.randn(2, 10))

    centered_eight = deepcopy(centered_eight)
    del centered_eight.sample_stats["log_likelihood"]
    with pytest.raises(TypeError):
        loo(centered_eight)


def test_loo_bad_scale(centered_eight):
    """Test loo with bad scale value."""
    with pytest.raises(TypeError):
        loo(centered_eight, scale="bad_scale")


def test_loo_bad_no_posterior_reff(centered_eight):
    loo(centered_eight, reff=None)
    centered_eight = deepcopy(centered_eight)
    del centered_eight.posterior
    with pytest.raises(TypeError):
        loo(centered_eight, reff=None)
    loo(centered_eight, reff=0.7)


def test_loo_warning(centered_eight):
    centered_eight = deepcopy(centered_eight)
    # make one of the khats infinity
    centered_eight.sample_stats["log_likelihood"][:, :, 1] = 10
    with pytest.warns(UserWarning) as records:
        assert loo(centered_eight, pointwise=True) is not None
    assert any("Estimated shape parameter" in str(record.message) for record in records)

    # make all of the khats infinity
    centered_eight.sample_stats["log_likelihood"][:, :, :] = 1
    with pytest.warns(UserWarning) as records:
        assert loo(centered_eight, pointwise=True) is not None
    assert any("Estimated shape parameter" in str(record.message) for record in records)


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_loo_print(centered_eight, scale):
    loo_data = loo(centered_eight, scale=scale, pointwise=False).__repr__()
    loo_pointwise = loo(centered_eight, scale=scale, pointwise=True).__repr__()
    assert loo_data is not None
    assert loo_pointwise is not None
    assert len(loo_data) < len(loo_pointwise)


def test_psislw(centered_eight):
    pareto_k = loo(centered_eight, pointwise=True, reff=0.7)["pareto_k"]
    log_likelihood = get_log_likelihood(centered_eight)
    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))
    assert_allclose(pareto_k, psislw(-log_likelihood, 0.7)[1])


def test_psislw_smooths_for_low_k():
    # check that log-weights are smoothed even when k < 1/3
    # https://github.com/arviz-devs/arviz/issues/2010
    rng = np.random.default_rng(44)
    x = rng.normal(size=100)
    x_smoothed, k = psislw(x.copy())
    assert k < 1 / 3
    assert not np.allclose(x - logsumexp(x), x_smoothed)


@pytest.mark.parametrize("probs", [True, False])
@pytest.mark.parametrize("kappa", [-1, -0.5, 1e-30, 0.5, 1])
@pytest.mark.parametrize("sigma", [0, 2])
def test_gpinv(probs, kappa, sigma):
    if probs:
        probs = np.array([0.1, 0.1, 0.1, 0.2, 0.3])
    else:
        probs = np.array([-0.1, 0.1, 0.1, 0.2, 0.3])
    assert len(_gpinv(probs, kappa, sigma)) == len(probs)


@pytest.mark.parametrize("func", [loo, waic])
def test_multidimensional_log_likelihood(func):
    llm = np.random.rand(4, 23, 15, 2)
    ll1 = llm.reshape(4, 23, 15 * 2)
    statsm = Dataset(dict(log_likelihood=DataArray(llm, dims=["chain", "draw", "a", "b"])))

    stats1 = Dataset(dict(log_likelihood=DataArray(ll1, dims=["chain", "draw", "v"])))

    post = Dataset(dict(mu=DataArray(np.random.rand(4, 23, 2), dims=["chain", "draw", "v"])))

    dsm = convert_to_inference_data(statsm, group="sample_stats")
    ds1 = convert_to_inference_data(stats1, group="sample_stats")
    dsp = convert_to_inference_data(post, group="posterior")

    dsm = concat(dsp, dsm)
    ds1 = concat(dsp, ds1)

    frm = func(dsm)
    fr1 = func(ds1)

    assert all(
        fr1[key] == frm[key] for key in fr1.index if key not in {"loo_i", "waic_i", "pareto_k"}
    )
    assert_array_almost_equal(frm[:4], fr1[:4])


@pytest.mark.parametrize(
    "args",
    [
        {"y": "obs"},
        {"y": "obs", "y_hat": "obs"},
        {"y": "arr", "y_hat": "obs"},
        {"y": "obs", "y_hat": "arr"},
        {"y": "arr", "y_hat": "arr"},
        {"y": "obs", "y_hat": "obs", "log_weights": "arr"},
        {"y": "arr", "y_hat": "obs", "log_weights": "arr"},
        {"y": "obs", "y_hat": "arr", "log_weights": "arr"},
        {"idata": False},
    ],
)
def test_loo_pit(centered_eight, args):
    y = args.get("y", None)
    y_hat = args.get("y_hat", None)
    log_weights = args.get("log_weights", None)
    y_arr = centered_eight.observed_data.obs
    y_hat_arr = centered_eight.posterior_predictive.obs.stack(__sample__=("chain", "draw"))
    log_like = get_log_likelihood(centered_eight).stack(__sample__=("chain", "draw"))
    n_samples = len(log_like.__sample__)
    ess_p = ess(centered_eight.posterior, method="mean")
    reff = np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples
    log_weights_arr = psislw(-log_like, reff=reff)[0]

    if args.get("idata", True):
        if y == "arr":
            y = y_arr
        if y_hat == "arr":
            y_hat = y_hat_arr
        if log_weights == "arr":
            log_weights = log_weights_arr
        loo_pit_data = loo_pit(idata=centered_eight, y=y, y_hat=y_hat, log_weights=log_weights)
    else:
        loo_pit_data = loo_pit(idata=None, y=y_arr, y_hat=y_hat_arr, log_weights=log_weights_arr)
    assert np.all((loo_pit_data >= 0) & (loo_pit_data <= 1))


@pytest.mark.parametrize(
    "args",
    [
        {"y": "y"},
        {"y": "y", "y_hat": "y"},
        {"y": "arr", "y_hat": "y"},
        {"y": "y", "y_hat": "arr"},
        {"y": "arr", "y_hat": "arr"},
        {"y": "y", "y_hat": "y", "log_weights": "arr"},
        {"y": "arr", "y_hat": "y", "log_weights": "arr"},
        {"y": "y", "y_hat": "arr", "log_weights": "arr"},
        {"idata": False},
    ],
)
def test_loo_pit_multidim(multidim_models, args):
    y = args.get("y", None)
    y_hat = args.get("y_hat", None)
    log_weights = args.get("log_weights", None)
    idata = multidim_models.model_1
    y_arr = idata.observed_data.y
    y_hat_arr = idata.posterior_predictive.y.stack(__sample__=("chain", "draw"))
    log_like = get_log_likelihood(idata).stack(__sample__=("chain", "draw"))
    n_samples = len(log_like.__sample__)
    ess_p = ess(idata.posterior, method="mean")
    reff = np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples
    log_weights_arr = psislw(-log_like, reff=reff)[0]

    if args.get("idata", True):
        if y == "arr":
            y = y_arr
        if y_hat == "arr":
            y_hat = y_hat_arr
        if log_weights == "arr":
            log_weights = log_weights_arr
        loo_pit_data = loo_pit(idata=idata, y=y, y_hat=y_hat, log_weights=log_weights)
    else:
        loo_pit_data = loo_pit(idata=None, y=y_arr, y_hat=y_hat_arr, log_weights=log_weights_arr)
    assert np.all((loo_pit_data >= 0) & (loo_pit_data <= 1))


def test_loo_pit_multi_lik():
    rng = np.random.default_rng(0)
    post_pred = rng.standard_normal(size=(4, 100, 10))
    obs = np.quantile(post_pred, np.linspace(0, 1, 10))
    obs[0] *= 0.9
    obs[-1] *= 1.1
    idata = from_dict(
        posterior={"a": np.random.randn(4, 100)},
        posterior_predictive={"y": post_pred},
        observed_data={"y": obs},
        log_likelihood={"y": -(post_pred**2), "decoy": np.zeros_like(post_pred)},
    )
    loo_pit_data = loo_pit(idata, y="y")
    assert np.all((loo_pit_data >= 0) & (loo_pit_data <= 1))


@pytest.mark.parametrize("input_type", ["idataarray", "idatanone_ystr", "yarr_yhatnone"])
def test_loo_pit_bad_input(centered_eight, input_type):
    """Test incompatible input combinations."""
    arr = np.random.random((8, 200))
    if input_type == "idataarray":
        with pytest.raises(ValueError, match=r"type InferenceData or None"):
            loo_pit(idata=arr, y="obs")
    elif input_type == "idatanone_ystr":
        with pytest.raises(ValueError, match=r"all 3.+must be array or DataArray"):
            loo_pit(idata=None, y="obs")
    elif input_type == "yarr_yhatnone":
        with pytest.raises(ValueError, match=r"y_hat.+None.+y.+str"):
            loo_pit(idata=centered_eight, y=arr, y_hat=None)


@pytest.mark.parametrize("arg", ["y", "y_hat", "log_weights"])
def test_loo_pit_bad_input_type(centered_eight, arg):
    """Test wrong input type (not None, str not DataArray."""
    kwargs = {"y": "obs", "y_hat": "obs", "log_weights": None}
    kwargs[arg] = 2  # use int instead of array-like
    with pytest.raises(ValueError, match=f"not {type(2)}"):
        loo_pit(idata=centered_eight, **kwargs)


@pytest.mark.parametrize("incompatibility", ["y-y_hat1", "y-y_hat2", "y_hat-log_weights"])
def test_loo_pit_bad_input_shape(incompatibility):
    """Test shape incompatibilities."""
    y = np.random.random(8)
    y_hat = np.random.random((8, 200))
    log_weights = np.random.random((8, 200))
    if incompatibility == "y-y_hat1":
        with pytest.raises(ValueError, match="1 more dimension"):
            loo_pit(y=y, y_hat=y_hat[None, :], log_weights=log_weights)
    elif incompatibility == "y-y_hat2":
        with pytest.raises(ValueError, match="y has shape"):
            loo_pit(y=y, y_hat=y_hat[1:3, :], log_weights=log_weights)
    elif incompatibility == "y_hat-log_weights":
        with pytest.raises(ValueError, match="must have the same shape"):
            loo_pit(y=y, y_hat=y_hat[:, :100], log_weights=log_weights)


@pytest.mark.parametrize("pointwise", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"group": "posterior_predictive", "var_names": {"posterior_predictive": "obs"}},
        {"group": "observed_data", "var_names": {"both": "obs"}, "out_data_shape": "shape"},
        {"var_names": {"both": "obs", "posterior": ["theta", "mu"]}},
        {"group": "observed_data", "out_name_data": "T_name"},
    ],
)
def test_apply_test_function(centered_eight, pointwise, inplace, kwargs):
    """Test some usual call cases of apply_test_function"""
    centered_eight = deepcopy(centered_eight)
    group = kwargs.get("group", "both")
    var_names = kwargs.get("var_names", None)
    out_data_shape = kwargs.get("out_data_shape", None)
    out_pp_shape = kwargs.get("out_pp_shape", None)
    out_name_data = kwargs.get("out_name_data", "T")
    if out_data_shape == "shape":
        out_data_shape = (8,) if pointwise else ()
    if out_pp_shape == "shape":
        out_pp_shape = (4, 500, 8) if pointwise else (4, 500)
    idata = deepcopy(centered_eight)
    idata_out = apply_test_function(
        idata,
        lambda y, theta: np.mean(y),
        group=group,
        var_names=var_names,
        pointwise=pointwise,
        out_name_data=out_name_data,
        out_data_shape=out_data_shape,
        out_pp_shape=out_pp_shape,
    )
    if inplace:
        assert idata is idata_out

    if group == "both":
        test_dict = {"observed_data": ["T"], "posterior_predictive": ["T"]}
    else:
        test_dict = {group: [kwargs.get("out_name_data", "T")]}

    fails = check_multiple_attrs(test_dict, idata_out)
    assert not fails


def test_apply_test_function_bad_group(centered_eight):
    """Test error when group is an invalid name."""
    with pytest.raises(ValueError, match="Invalid group argument"):
        apply_test_function(centered_eight, lambda y, theta: y, group="bad_group")


def test_apply_test_function_missing_group():
    """Test error when InferenceData object is missing a required group.

    The function cannot work if group="both" but InferenceData object has no
    posterior_predictive group.
    """
    idata = from_dict(
        posterior={"a": np.random.random((4, 500, 30))}, observed_data={"y": np.random.random(30)}
    )
    with pytest.raises(ValueError, match="must have posterior_predictive"):
        apply_test_function(idata, lambda y, theta: np.mean, group="both")


def test_apply_test_function_should_overwrite_error(centered_eight):
    """Test error when overwrite=False but out_name is already a present variable."""
    with pytest.raises(ValueError, match="Should overwrite"):
        apply_test_function(centered_eight, lambda y, theta: y, out_name_data="obs")
