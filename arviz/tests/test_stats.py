# pylint: disable=redefined-outer-name, no-member
from copy import deepcopy
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
import pytest
from scipy.stats import linregress
from xarray import Dataset, DataArray


from ..data import load_arviz_data, from_dict, convert_to_inference_data, concat
from ..stats import (
    compare,
    hpd,
    loo,
    r2_score,
    waic,
    psislw,
    summary,
    loo_pit,
    ess,
    apply_test_function,
)
from ..stats.stats import _gpinv
from ..utils import Numba
from .helpers import check_multiple_attrs, multidim_models  # pylint: disable=unused-import
from ..rcparams import rcParams


rcParams["data.load"] = "eager"


@pytest.fixture(scope="session")
def centered_eight():
    centered_eight = load_arviz_data("centered_eight")
    return centered_eight


@pytest.fixture(scope="session")
def non_centered_eight():
    non_centered_eight = load_arviz_data("non_centered_eight")
    return non_centered_eight


def test_hpd():
    normal_sample = np.random.randn(5000000)
    interval = hpd(normal_sample)
    assert_array_almost_equal(interval, [-1.88, 1.88], 2)


def test_hpd_multimodal():
    normal_sample = np.concatenate(
        (np.random.normal(-4, 1, 2500000), np.random.normal(2, 0.5, 2500000))
    )
    intervals = hpd(normal_sample, multimodal=True)
    assert_array_almost_equal(intervals, [[-5.8, -2.2], [0.9, 3.1]], 1)


def test_hpd_circular():
    normal_sample = np.random.vonmises(np.pi, 1, 5000000)
    interval = hpd(normal_sample, circular=True)
    assert_array_almost_equal(interval, [0.6, -0.6], 1)


def test_hpd_bad_ci():
    normal_sample = np.random.randn(10)
    with pytest.raises(ValueError):
        hpd(normal_sample, credible_interval=2)


def test_r2_score():
    x = np.linspace(0, 1, 100)
    y = np.random.normal(x, 1)
    res = linregress(x, y)
    assert_allclose(res.rvalue ** 2, r2_score(y, res.intercept + res.slope * x).r2, 2)


def test_r2_score_multivariate():
    x = np.linspace(0, 1, 100)
    y = np.random.normal(x, 1)
    res = linregress(x, y)
    y_multivariate = np.c_[y, y]
    y_multivariate_pred = np.c_[res.intercept + res.slope * x, res.intercept + res.slope * x]
    assert not np.isnan(r2_score(y_multivariate, y_multivariate_pred).r2)


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
    with pytest.raises(NotImplementedError):
        compare(model_dict, ic="Unknown", method="stacking")
    with pytest.raises(ValueError):
        compare(model_dict, ic="loo", method="Unknown")


@pytest.mark.parametrize("ic", ["waic", "loo"])
@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
@pytest.mark.parametrize("scale", ["deviance", "log", "negative_log"])
def test_compare_different(centered_eight, non_centered_eight, ic, method, scale):
    model_dict = {"centered": centered_eight, "non_centered": non_centered_eight}
    weight = compare(model_dict, ic=ic, method=method, scale=scale)["weight"]
    assert weight["non_centered"] >= weight["centered"]
    assert_allclose(np.sum(weight), 1.0)


@pytest.mark.parametrize("ic", ["waic", "loo"])
@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_compare_different_multidim(multidim_models, ic, method):
    model_dict = {"model_1": multidim_models.model_1, "model_2": multidim_models.model_2}
    weight = compare(model_dict, ic=ic, method=method)["weight"]

    # this should hold because the same seed is always used
    assert weight["model_1"] >= weight["model_2"]
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


@pytest.mark.parametrize("var_names_expected", ((None, 10), ("mu", 1), (["mu", "tau"], 2)))
def test_summary_var_names(centered_eight, var_names_expected):
    var_names, expected = var_names_expected
    summary_df = summary(centered_eight, var_names=var_names)
    assert len(summary_df.index) == expected


@pytest.mark.parametrize("include_circ", [True, False])
def test_summary_include_circ(centered_eight, include_circ):
    assert summary(centered_eight, include_circ=include_circ) is not None
    state = Numba.numba_flag
    Numba.disable_numba()
    assert summary(centered_eight, include_circ=include_circ) is not NotImplementedError
    Numba.enable_numba()
    assert state == Numba.numba_flag


METRICS_NAMES = [
    "mean",
    "sd",
    "hpd_3%",
    "hpd_97%",
    "mcse_mean",
    "mcse_sd",
    "ess_mean",
    "ess_sd",
    "ess_bulk",
    "ess_tail",
    "r_hat",
]


@pytest.mark.parametrize(
    "params",
    (("all", METRICS_NAMES), ("stats", METRICS_NAMES[:4]), ("diagnostics", METRICS_NAMES[4:])),
)
def test_summary_kind(centered_eight, params):
    kind, metrics_names_ = params
    summary_df = summary(centered_eight, kind=kind)
    assert_array_equal(summary_df.columns, metrics_names_)


@pytest.mark.parametrize("fmt", ["wide", "long", "xarray"])
def test_summary_fmt(centered_eight, fmt):
    assert summary(centered_eight, fmt=fmt) is not None


@pytest.mark.parametrize("order", ["C", "F"])
def test_summary_unpack_order(order):
    data = from_dict({"a": np.random.randn(4, 100, 4, 5, 3)})
    az_summary = summary(data, order=order, fmt="wide")
    assert az_summary is not None
    if order != "F":
        first_index = 4
        second_index = 5
        third_index = 3
    else:
        first_index = 3
        second_index = 5
        third_index = 4
    column_order = []
    for idx1 in range(first_index):
        for idx2 in range(second_index):
            for idx3 in range(third_index):
                if order != "F":
                    column_order.append("a[{},{},{}]".format(idx1, idx2, idx3))
                else:
                    column_order.append("a[{},{},{}]".format(idx3, idx2, idx1))
    for col1, col2 in zip(list(az_summary.index), column_order):
        assert col1 == col2


@pytest.mark.parametrize("origin", [0, 1, 2, 3])
def test_summary_index_origin(origin):
    data = from_dict({"a": np.random.randn(2, 50, 10)})
    az_summary = summary(data, index_origin=origin, fmt="wide")
    assert az_summary is not None
    for i, col in enumerate(list(az_summary.index)):
        assert col == "a[{}]".format(i + origin)


@pytest.mark.parametrize(
    "stat_funcs", [[np.var], {"var": np.var, "var2": lambda x: np.var(x) ** 2}]
)
def test_summary_stat_func(centered_eight, stat_funcs):
    arviz_summary = summary(centered_eight, stat_funcs=stat_funcs)
    assert arviz_summary is not None
    assert hasattr(arviz_summary, "var")


def test_summary_nan(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior.theta[:, :, 0] = np.nan
    summary_xarray = summary(centered_eight)
    assert summary_xarray is not None
    assert summary_xarray.loc["theta[0]"].isnull().all()
    assert (
        summary_xarray.loc[[ix for ix in summary_xarray.index if ix != "theta[0]"]]
        .notnull()
        .all()
        .all()
    )


@pytest.mark.parametrize("fmt", [1, "bad_fmt"])
def test_summary_bad_fmt(centered_eight, fmt):
    with pytest.raises(TypeError):
        summary(centered_eight, fmt=fmt)


@pytest.mark.parametrize("order", [1, "bad_order"])
def test_summary_bad_unpack_order(centered_eight, order):
    with pytest.raises(TypeError):
        summary(centered_eight, order=order)


@pytest.mark.parametrize("scale", ["deviance", "log", "negative_log"])
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


@pytest.mark.parametrize("scale", ["deviance", "log", "negative_log"])
def test_waic_print(centered_eight, scale):
    waic_data = waic(centered_eight, scale=scale).__repr__()
    waic_pointwise = waic(centered_eight, scale=scale, pointwise=True).__repr__()
    assert waic_data is not None
    assert waic_pointwise is not None
    assert waic_data == waic_pointwise


@pytest.mark.parametrize("scale", ["deviance", "log", "negative_log"])
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
    with pytest.warns(UserWarning) as record:
        assert loo(centered_eight, pointwise=True) is not None
    assert len(record) == 1
    assert "Estimated shape parameter" in str(record[0].message)
    # make all of the khats infinity
    centered_eight.sample_stats["log_likelihood"][:, :, :] = 1
    with pytest.warns(UserWarning) as record:
        assert loo(centered_eight, pointwise=True) is not None
    assert len(record) == 1
    assert "Estimated shape parameter" in str(record[0].message)


@pytest.mark.parametrize("scale", ["deviance", "log", "negative_log"])
def test_loo_print(centered_eight, scale):
    loo_data = loo(centered_eight, scale=scale).__repr__()
    loo_pointwise = loo(centered_eight, scale=scale, pointwise=True).__repr__()
    assert loo_data is not None
    assert loo_pointwise is not None
    assert len(loo_data) < len(loo_pointwise)
    assert loo_data == loo_pointwise[: len(loo_data)]


def test_psislw(centered_eight):
    pareto_k = loo(centered_eight, pointwise=True, reff=0.7)["pareto_k"]
    log_likelihood = centered_eight.sample_stats.log_likelihood  # pylint: disable=no-member
    log_likelihood = log_likelihood.stack(sample=("chain", "draw"))
    assert_allclose(pareto_k, psislw(-log_likelihood, 0.7)[1])


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

    assert (fr1 == frm).all()
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
    y_hat_arr = centered_eight.posterior_predictive.obs.stack(sample=("chain", "draw"))
    log_like = centered_eight.sample_stats.log_likelihood.stack(sample=("chain", "draw"))
    n_samples = len(log_like.sample)
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
    y_hat_arr = idata.posterior_predictive.y.stack(sample=("chain", "draw"))
    log_like = idata.sample_stats.log_likelihood.stack(sample=("chain", "draw"))
    n_samples = len(log_like.sample)
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
    with pytest.raises(ValueError, match="not {}".format(type(2))):
        loo_pit(idata=centered_eight, **kwargs)


@pytest.mark.parametrize("incompatibility", ["y-y_hat1", "y-y_hat2", "y_hat-log_weights"])
def test_loo_pit_bad_input_shape(incompatibility):
    """Test shape incompatiblities."""
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


def test_numba_stats():
    """Numba test for r2_score"""
    state = Numba.numba_flag  # Store the current state of Numba
    set_1 = np.random.randn(100, 100)
    set_2 = np.random.randn(100, 100)
    set_3 = np.random.rand(100)
    set_4 = np.random.rand(100)
    Numba.disable_numba()
    non_numba = r2_score(set_1, set_2)
    non_numba_one_dimensional = r2_score(set_3, set_4)
    Numba.enable_numba()
    with_numba = r2_score(set_1, set_2)
    with_numba_one_dimensional = r2_score(set_3, set_4)
    assert state == Numba.numba_flag  # Ensure that inital state = final state
    assert np.allclose(non_numba, with_numba)
    assert np.allclose(non_numba_one_dimensional, with_numba_one_dimensional)
