# pylint: disable=redefined-outer-name
from copy import deepcopy
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import pytest
from scipy.stats import linregress
from xarray import Dataset, DataArray


from ..data import load_arviz_data, from_dict, convert_to_inference_data, concat
from ..stats import compare, hpd, loo, r2_score, waic, psislw, summary
from ..stats.stats import _gpinv


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
def test_compare_same(centered_eight, method):
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
def test_summary_var_names(var_names_expected):
    var_names, expected = var_names_expected
    centered = load_arviz_data("centered_eight")
    summary_df = summary(centered, var_names=var_names)
    assert len(summary_df.index) == expected


@pytest.mark.parametrize("include_circ", [True, False])
def test_summary_include_circ(centered_eight, include_circ):
    assert summary(centered_eight, include_circ=include_circ) is not None


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
def test_waic(centered_eight, scale):
    """Test widely available information criterion calculation"""
    assert waic(centered_eight, scale=scale) is not None
    assert waic(centered_eight, pointwise=True, scale=scale) is not None


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


def test_loo(centered_eight):
    assert loo(centered_eight) is not None


def test_loo_one_chain(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.posterior = centered_eight.posterior.drop([1, 2, 3], "chain")
    centered_eight.sample_stats = centered_eight.sample_stats.drop([1, 2, 3], "chain")
    assert loo(centered_eight) is not None


@pytest.mark.parametrize("scale", ["deviance", "log", "negative_log"])
def test_loo_pointwise(centered_eight, scale):
    """Test pointwise loo with different scales."""
    loo_results = loo(centered_eight, scale=scale, pointwise=True)
    assert loo_results is not None
    assert hasattr(loo_results, "loo_scale")
    assert hasattr(loo_results, "pareto_k")
    assert hasattr(loo_results, "loo_i")


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


def test_loo_warning(centered_eight):
    centered_eight = deepcopy(centered_eight)
    # make one of the khats infinity
    centered_eight.sample_stats["log_likelihood"][:, :, 1] = 10
    with pytest.warns(UserWarning):
        assert loo(centered_eight, pointwise=True) is not None
    # make all of the khats infinity
    centered_eight.sample_stats["log_likelihood"][:, :, :] = 0
    with pytest.warns(UserWarning):
        assert loo(centered_eight, pointwise=True) is not None


@pytest.mark.parametrize("scale", ["deviance", "log", "negative_log"])
def test_loo_print(centered_eight, scale):
    loo_data = loo(centered_eight, scale=scale).__repr__()
    loo_pointwise = loo(centered_eight, scale=scale, pointwise=True).__repr__()
    assert loo_data is not None
    assert loo_pointwise is not None
    assert len(loo_data) < len(loo_pointwise)
    assert loo_data == loo_pointwise[: len(loo_data)]


def test_psislw():
    data = load_arviz_data("centered_eight")
    pareto_k = loo(data, pointwise=True, reff=0.7)["pareto_k"]
    log_likelihood = data.sample_stats.log_likelihood  # pylint: disable=no-member
    log_likelihood = log_likelihood.stack(samples=("chain", "draw"))
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
    np.random.seed(17)
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
