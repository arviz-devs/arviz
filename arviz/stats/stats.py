# pylint: disable=too-many-lines
"""Statistical functions in ArviZ."""

import itertools
import warnings
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Mapping, cast, Callable

import numpy as np
import pandas as pd
import scipy.stats as st
from xarray_einstats import stats
import xarray as xr
from scipy.optimize import minimize
from typing_extensions import Literal

NO_GET_ARGS: bool = False
try:
    from typing_extensions import get_args
except ImportError:
    NO_GET_ARGS = True

from .. import _log
from ..data import InferenceData, convert_to_dataset, convert_to_inference_data, extract
from ..rcparams import rcParams, ScaleKeyword, ICKeyword
from ..utils import Numba, _numba_var, _var_names, get_coords
from .density_utils import get_bins as _get_bins
from .density_utils import histogram as _histogram
from .density_utils import kde as _kde
from .diagnostics import _mc_error, _multichain_statistics, ess
from .stats_utils import ELPDData, _circular_standard_deviation, smooth_data
from .stats_utils import get_log_likelihood as _get_log_likelihood
from .stats_utils import logsumexp as _logsumexp
from .stats_utils import make_ufunc as _make_ufunc
from .stats_utils import stats_variance_2d as svar
from .stats_utils import wrap_xarray_ufunc as _wrap_xarray_ufunc
from ..sel_utils import xarray_var_iter
from ..labels import BaseLabeller


__all__ = [
    "apply_test_function",
    "compare",
    "hdi",
    "loo",
    "loo_pit",
    "psislw",
    "r2_samples",
    "r2_score",
    "summary",
    "waic",
    "weight_predictions",
    "_calculate_ics",
]


def compare(
    compare_dict: Mapping[str, InferenceData],
    ic: Optional[ICKeyword] = None,
    method: Literal["stacking", "BB-pseudo-BMA", "pseudo-BMA"] = "stacking",
    b_samples: int = 1000,
    alpha: float = 1,
    seed=None,
    scale: Optional[ScaleKeyword] = None,
    var_name: Optional[str] = None,
):
    r"""Compare models based on  their expected log pointwise predictive density (ELPD).

    The ELPD is estimated either by Pareto smoothed importance sampling leave-one-out
    cross-validation (LOO) or using the widely applicable information criterion (WAIC).
    We recommend loo. Read more theory here - in a paper by some of the
    leading authorities on model comparison dx.doi.org/10.1111/1467-9868.00353

    Parameters
    ----------
    compare_dict: dict of {str: InferenceData or ELPDData}
        A dictionary of model names and :class:`arviz.InferenceData` or ``ELPDData``.
    ic: str, optional
        Method to estimate the ELPD, available options are "loo" or "waic". Defaults to
        ``rcParams["stats.information_criterion"]``.
    method: str, optional
        Method used to estimate the weights for each model. Available options are:

        - 'stacking' : stacking of predictive distributions.
        - 'BB-pseudo-BMA' : pseudo-Bayesian Model averaging using Akaike-type
          weighting. The weights are stabilized using the Bayesian bootstrap.
        - 'pseudo-BMA': pseudo-Bayesian Model averaging using Akaike-type
          weighting, without Bootstrap stabilization (not recommended).

        For more information read https://arxiv.org/abs/1704.02030
    b_samples: int, optional default = 1000
        Number of samples taken by the Bayesian bootstrap estimation.
        Only useful when method = 'BB-pseudo-BMA'.
        Defaults to ``rcParams["stats.ic_compare_method"]``.
    alpha: float, optional
        The shape parameter in the Dirichlet distribution used for the Bayesian bootstrap. Only
        useful when method = 'BB-pseudo-BMA'. When alpha=1 (default), the distribution is uniform
        on the simplex. A smaller alpha will keeps the final weights more away from 0 and 1.
    seed: int or np.random.RandomState instance, optional
        If int or RandomState, use it for seeding Bayesian bootstrap. Only
        useful when method = 'BB-pseudo-BMA'. Default None the global
        :mod:`numpy.random` state is used.
    scale: str, optional
        Output scale for IC. Available options are:

        - `log` : (default) log-score (after Vehtari et al. (2017))
        - `negative_log` : -1 * (log-score)
        - `deviance` : -2 * (log-score)

        A higher log-score (or a lower deviance) indicates a model with better predictive
        accuracy.
    var_name: str, optional
        If there is more than a single observed variable in the ``InferenceData``, which
        should be used as the basis for comparison.

    Returns
    -------
    A DataFrame, ordered from best to worst model (measured by the ELPD).
    The index reflects the key with which the models are passed to this function. The columns are:
    rank: The rank-order of the models. 0 is the best.
    elpd: ELPD estimated either using (PSIS-LOO-CV `elpd_loo` or WAIC `elpd_waic`).
        Higher ELPD indicates higher out-of-sample predictive fit ("better" model).
        If `scale` is `deviance` or `negative_log` smaller values indicates
        higher out-of-sample predictive fit ("better" model).
    pIC: Estimated effective number of parameters.
    elpd_diff: The difference in ELPD between two models.
        If more than two models are compared, the difference is computed relative to the
        top-ranked model, that always has a elpd_diff of 0.
    weight: Relative weight for each model.
        This can be loosely interpreted as the probability of each model (among the compared model)
        given the data. By default the uncertainty in the weights estimation is considered using
        Bayesian bootstrap.
    SE: Standard error of the ELPD estimate.
        If method = BB-pseudo-BMA these values are estimated using Bayesian bootstrap.
    dSE: Standard error of the difference in ELPD between each model and the top-ranked model.
        It's always 0 for the top-ranked model.
    warning: A value of 1 indicates that the computation of the ELPD may not be reliable.
        This could be indication of WAIC/LOO starting to fail see
        http://arxiv.org/abs/1507.04544 for details.
    scale: Scale used for the ELPD.

    Examples
    --------
    Compare the centered and non centered models of the eight school problem:

    .. ipython::

        In [1]: import arviz as az
           ...: data1 = az.load_arviz_data("non_centered_eight")
           ...: data2 = az.load_arviz_data("centered_eight")
           ...: compare_dict = {"non centered": data1, "centered": data2}
           ...: az.compare(compare_dict)

    Compare the models using PSIS-LOO-CV, returning the ELPD in log scale and calculating the
    weights using the stacking method.

    .. ipython::

        In [1]: az.compare(compare_dict, ic="loo", method="stacking", scale="log")

    See Also
    --------
    loo :
        Compute the ELPD using the Pareto smoothed importance sampling Leave-one-out
        cross-validation method.
    waic : Compute the ELPD using the widely applicable information criterion.
    plot_compare : Summary plot for model comparison.

    References
    ----------
    .. [1] Vehtari, A., Gelman, A. & Gabry, J. Practical Bayesian model evaluation using
        leave-one-out cross-validation and WAIC. Stat Comput 27, 1413–1432 (2017)
        see https://doi.org/10.1007/s11222-016-9696-4

    """
    try:
        (ics_dict, scale, ic) = _calculate_ics(compare_dict, scale=scale, ic=ic, var_name=var_name)
    except Exception as e:
        raise e.__class__("Encountered error in ELPD computation of compare.") from e
    names = list(ics_dict.keys())
    if ic == "loo":
        df_comp = pd.DataFrame(
            index=names,
            columns=[
                "rank",
                "elpd_loo",
                "p_loo",
                "elpd_diff",
                "weight",
                "se",
                "dse",
                "warning",
                "scale",
            ],
            dtype=np.float_,
        )
    elif ic == "waic":
        df_comp = pd.DataFrame(
            index=names,
            columns=[
                "rank",
                "elpd_waic",
                "p_waic",
                "elpd_diff",
                "weight",
                "se",
                "dse",
                "warning",
                "scale",
            ],
            dtype=np.float_,
        )
    else:
        raise NotImplementedError(f"The information criterion {ic} is not supported.")

    if scale == "log":
        scale_value = 1
        ascending = False
    else:
        if scale == "negative_log":
            scale_value = -1
        else:
            scale_value = -2
        ascending = True

    method = rcParams["stats.ic_compare_method"] if method is None else method
    if method.lower() not in ["stacking", "bb-pseudo-bma", "pseudo-bma"]:
        raise ValueError(f"The method {method}, to compute weights, is not supported.")

    p_ic = f"p_{ic}"
    ic_i = f"{ic}_i"

    ics = pd.DataFrame.from_dict(ics_dict, orient="index")
    ics.sort_values(by=f"elpd_{ic}", inplace=True, ascending=ascending)
    ics[ic_i] = ics[ic_i].apply(lambda x: x.values.flatten())

    if method.lower() == "stacking":
        rows, cols, ic_i_val = _ic_matrix(ics, ic_i)
        exp_ic_i = np.exp(ic_i_val / scale_value)
        km1 = cols - 1

        def w_fuller(weights):
            return np.concatenate((weights, [max(1.0 - np.sum(weights), 0.0)]))

        def log_score(weights):
            w_full = w_fuller(weights)
            score = 0.0
            for i in range(rows):
                score += np.log(np.dot(exp_ic_i[i], w_full))
            return -score

        def gradient(weights):
            w_full = w_fuller(weights)
            grad = np.zeros(km1)
            for k, i in itertools.product(range(km1), range(rows)):
                grad[k] += (exp_ic_i[i, k] - exp_ic_i[i, km1]) / np.dot(exp_ic_i[i], w_full)
            return -grad

        theta = np.full(km1, 1.0 / cols)
        bounds = [(0.0, 1.0) for _ in range(km1)]
        constraints = [
            {"type": "ineq", "fun": lambda x: -np.sum(x) + 1.0},
            {"type": "ineq", "fun": np.sum},
        ]

        weights = minimize(
            fun=log_score, x0=theta, jac=gradient, bounds=bounds, constraints=constraints
        )

        weights = w_fuller(weights["x"])
        ses = ics["se"]

    elif method.lower() == "bb-pseudo-bma":
        rows, cols, ic_i_val = _ic_matrix(ics, ic_i)
        ic_i_val = ic_i_val * rows

        b_weighting = st.dirichlet.rvs(alpha=[alpha] * rows, size=b_samples, random_state=seed)
        weights = np.zeros((b_samples, cols))
        z_bs = np.zeros_like(weights)
        for i in range(b_samples):
            z_b = np.dot(b_weighting[i], ic_i_val)
            u_weights = np.exp((z_b - np.max(z_b)) / scale_value)
            z_bs[i] = z_b  # pylint: disable=unsupported-assignment-operation
            weights[i] = u_weights / np.sum(u_weights)

        weights = weights.mean(axis=0)
        ses = pd.Series(z_bs.std(axis=0), index=names)  # pylint: disable=no-member

    elif method.lower() == "pseudo-bma":
        min_ic = ics.iloc[0][f"elpd_{ic}"]
        z_rv = np.exp((ics[f"elpd_{ic}"] - min_ic) / scale_value)
        weights = z_rv / np.sum(z_rv)
        ses = ics["se"]

    if np.any(weights):
        min_ic_i_val = ics[ic_i].iloc[0]
        for idx, val in enumerate(ics.index):
            res = ics.loc[val]
            if scale_value < 0:
                diff = res[ic_i] - min_ic_i_val
            else:
                diff = min_ic_i_val - res[ic_i]
            d_ic = np.sum(diff)
            d_std_err = np.sqrt(len(diff) * np.var(diff))
            std_err = ses.loc[val]
            weight = weights[idx]
            df_comp.loc[val] = (
                idx,
                res[f"elpd_{ic}"],
                res[p_ic],
                d_ic,
                weight,
                std_err,
                d_std_err,
                res["warning"],
                res["scale"],
            )

    df_comp["rank"] = df_comp["rank"].astype(int)
    df_comp["warning"] = df_comp["warning"].astype(bool)
    return df_comp.sort_values(by=f"elpd_{ic}", ascending=ascending)


def _ic_matrix(ics, ic_i):
    """Store the previously computed pointwise predictive accuracy values (ics) in a 2D matrix."""
    cols, _ = ics.shape
    rows = len(ics[ic_i].iloc[0])
    ic_i_val = np.zeros((rows, cols))

    for idx, val in enumerate(ics.index):
        ic = ics.loc[val][ic_i]

        if len(ic) != rows:
            raise ValueError("The number of observations should be the same across all models")

        ic_i_val[:, idx] = ic

    return rows, cols, ic_i_val


def _calculate_ics(
    compare_dict,
    scale: Optional[ScaleKeyword] = None,
    ic: Optional[ICKeyword] = None,
    var_name: Optional[str] = None,
):
    """Calculate LOO or WAIC only if necessary.

    It always calls the ic function with ``pointwise=True``.

    Parameters
    ----------
    compare_dict :  dict of {str : InferenceData or ELPDData}
        A dictionary of model names and InferenceData or ELPDData objects
    scale : str, optional
        Output scale for IC. Available options are:

        - `log` : (default) log-score (after Vehtari et al. (2017))
        - `negative_log` : -1 * (log-score)
        - `deviance` : -2 * (log-score)

        A higher log-score (or a lower deviance) indicates a model with better predictive accuracy.
    ic : str, optional
        Information Criterion (PSIS-LOO `loo` or WAIC `waic`) used to compare models.
        Defaults to ``rcParams["stats.information_criterion"]``.
    var_name : str, optional
        Name of the variable storing pointwise log likelihood values in ``log_likelihood`` group.


    Returns
    -------
    compare_dict : dict of ELPDData
    scale : str
    ic : str

    """
    precomputed_elpds = {
        name: elpd_data
        for name, elpd_data in compare_dict.items()
        if isinstance(elpd_data, ELPDData)
    }
    precomputed_ic = None
    precomputed_scale = None
    if precomputed_elpds:
        _, arbitrary_elpd = precomputed_elpds.popitem()
        precomputed_ic = arbitrary_elpd.index[0].split("_")[1]
        precomputed_scale = arbitrary_elpd["scale"]
        raise_non_pointwise = f"{precomputed_ic}_i" not in arbitrary_elpd
        if any(
            elpd_data.index[0].split("_")[1] != precomputed_ic
            for elpd_data in precomputed_elpds.values()
        ):
            raise ValueError(
                "All information criteria to be compared must be the same "
                "but found both loo and waic."
            )
        if any(elpd_data["scale"] != precomputed_scale for elpd_data in precomputed_elpds.values()):
            raise ValueError("All information criteria to be compared must use the same scale")
        if (
            any(f"{precomputed_ic}_i" not in elpd_data for elpd_data in precomputed_elpds.values())
            or raise_non_pointwise
        ):
            raise ValueError("Not all provided ELPDData have been calculated with pointwise=True")
        if ic is not None and ic.lower() != precomputed_ic:
            warnings.warn(
                "Provided ic argument is incompatible with precomputed elpd data. "
                f"Using ic from precomputed elpddata: {precomputed_ic}"
            )
            ic = precomputed_ic
        if scale is not None and scale.lower() != precomputed_scale:
            warnings.warn(
                "Provided scale argument is incompatible with precomputed elpd data. "
                f"Using scale from precomputed elpddata: {precomputed_scale}"
            )
            scale = precomputed_scale

    if ic is None and precomputed_ic is None:
        ic = cast(ICKeyword, rcParams["stats.information_criterion"])
    elif ic is None:
        ic = precomputed_ic
    else:
        ic = cast(ICKeyword, ic.lower())
    allowable = ["loo", "waic"] if NO_GET_ARGS else get_args(ICKeyword)
    if ic not in allowable:
        raise ValueError(f"{ic} is not a valid value for ic: must be in {allowable}")

    if scale is None and precomputed_scale is None:
        scale = cast(ScaleKeyword, rcParams["stats.ic_scale"])
    elif scale is None:
        scale = precomputed_scale
    else:
        scale = cast(ScaleKeyword, scale.lower())
    allowable = ["log", "negative_log", "deviance"] if NO_GET_ARGS else get_args(ScaleKeyword)
    if scale not in allowable:
        raise ValueError(f"{scale} is not a valid value for scale: must be in {allowable}")

    if ic == "loo":
        ic_func: Callable = loo
    elif ic == "waic":
        ic_func = waic
    else:
        raise NotImplementedError(f"The information criterion {ic} is not supported.")

    compare_dict = deepcopy(compare_dict)
    for name, dataset in compare_dict.items():
        if not isinstance(dataset, ELPDData):
            try:
                compare_dict[name] = ic_func(
                    convert_to_inference_data(dataset),
                    pointwise=True,
                    scale=scale,
                    var_name=var_name,
                )
            except Exception as e:
                raise e.__class__(
                    f"Encountered error trying to compute {ic} from model {name}."
                ) from e
    return (compare_dict, scale, ic)


def hdi(
    ary,
    hdi_prob=None,
    circular=False,
    multimodal=False,
    skipna=False,
    group="posterior",
    var_names=None,
    filter_vars=None,
    coords=None,
    max_modes=10,
    dask_kwargs=None,
    **kwargs,
):
    """
    Calculate highest density interval (HDI) of array for given probability.

    The HDI is the minimum width Bayesian credible interval (BCI).

    Parameters
    ----------
    ary: obj
        object containing posterior samples.
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
    hdi_prob: float, optional
        Prob for which the highest density interval will be computed. Defaults to
        ``stats.hdi_prob`` rcParam.
    circular: bool, optional
        Whether to compute the hdi taking into account `x` is a circular variable
        (in the range [-np.pi, np.pi]) or not. Defaults to False (i.e non-circular variables).
        Only works if multimodal is False.
    multimodal: bool, optional
        If true it may compute more than one hdi if the distribution is multimodal and the
        modes are well separated.
    skipna: bool, optional
        If true ignores nan values when computing the hdi. Defaults to false.
    group: str, optional
        Specifies which InferenceData group should be used to calculate hdi.
        Defaults to 'posterior'
    var_names: list, optional
        Names of variables to include in the hdi report. Prefix the variables by ``~``
        when you want to exclude them from the report: `["~beta"]` instead of `["beta"]`
        (see :func:`arviz.summary` for more details).
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        ``pandas.filter``.
    coords: mapping, optional
        Specifies the subset over to calculate hdi.
    max_modes: int, optional
        Specifies the maximum number of modes for multimodal case.
    dask_kwargs : dict, optional
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.
    kwargs: dict, optional
        Additional keywords passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    np.ndarray or xarray.Dataset, depending upon input
        lower(s) and upper(s) values of the interval(s).

    See Also
    --------
    plot_hdi : Plot highest density intervals for regression data.
    xarray.Dataset.quantile : Calculate quantiles of array for given probabilities.

    Examples
    --------
    Calculate the HDI of a Normal random variable:

    .. ipython::

        In [1]: import arviz as az
           ...: import numpy as np
           ...: data = np.random.normal(size=2000)
           ...: az.hdi(data, hdi_prob=.68)

    Calculate the HDI of a dataset:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data('centered_eight')
           ...: az.hdi(data)

    We can also calculate the HDI of some of the variables of dataset:

    .. ipython::

        In [1]: az.hdi(data, var_names=["mu", "theta"])

    By default, ``hdi`` is calculated over the ``chain`` and ``draw`` dimensions. We can use the
    ``input_core_dims`` argument of :func:`~arviz.wrap_xarray_ufunc` to change this. In this example
    we calculate the HDI also over the ``school`` dimension:

    .. ipython::

        In [1]: az.hdi(data, var_names="theta", input_core_dims = [["chain","draw", "school"]])

    We can also calculate the hdi over a particular selection:

    .. ipython::

        In [1]: az.hdi(data, coords={"chain":[0, 1, 3]}, input_core_dims = [["draw"]])

    """
    if hdi_prob is None:
        hdi_prob = rcParams["stats.hdi_prob"]
    elif not 1 >= hdi_prob > 0:
        raise ValueError("The value of hdi_prob should be in the interval (0, 1]")

    func_kwargs = {
        "hdi_prob": hdi_prob,
        "skipna": skipna,
        "out_shape": (max_modes, 2) if multimodal else (2,),
    }
    kwargs.setdefault("output_core_dims", [["mode", "hdi"] if multimodal else ["hdi"]])
    if not multimodal:
        func_kwargs["circular"] = circular
    else:
        func_kwargs["max_modes"] = max_modes

    func = _hdi_multimodal if multimodal else _hdi

    isarray = isinstance(ary, np.ndarray)
    if isarray and ary.ndim <= 1:
        func_kwargs.pop("out_shape")
        hdi_data = func(ary, **func_kwargs)  # pylint: disable=unexpected-keyword-arg
        return hdi_data[~np.isnan(hdi_data).all(axis=1), :] if multimodal else hdi_data

    if isarray and ary.ndim == 2:
        warnings.warn(
            "hdi currently interprets 2d data as (draw, shape) but this will change in "
            "a future release to (chain, draw) for coherence with other functions",
            FutureWarning,
            stacklevel=2,
        )
        ary = np.expand_dims(ary, 0)

    ary = convert_to_dataset(ary, group=group)
    if coords is not None:
        ary = get_coords(ary, coords)
    var_names = _var_names(var_names, ary, filter_vars)
    ary = ary[var_names] if var_names else ary

    hdi_coord = xr.DataArray(["lower", "higher"], dims=["hdi"], attrs=dict(hdi_prob=hdi_prob))
    hdi_data = _wrap_xarray_ufunc(
        func, ary, func_kwargs=func_kwargs, dask_kwargs=dask_kwargs, **kwargs
    ).assign_coords({"hdi": hdi_coord})
    hdi_data = hdi_data.dropna("mode", how="all") if multimodal else hdi_data
    return hdi_data.x.values if isarray else hdi_data


def _hdi(ary, hdi_prob, circular, skipna):
    """Compute hpi over the flattened array."""
    ary = ary.flatten()
    if skipna:
        nans = np.isnan(ary)
        if not nans.all():
            ary = ary[~nans]
    n = len(ary)

    if circular:
        mean = st.circmean(ary, high=np.pi, low=-np.pi)
        ary = ary - mean
        ary = np.arctan2(np.sin(ary), np.cos(ary))

    ary = np.sort(ary)
    interval_idx_inc = int(np.floor(hdi_prob * n))
    n_intervals = n - interval_idx_inc
    interval_width = np.subtract(ary[interval_idx_inc:], ary[:n_intervals], dtype=np.float_)

    if len(interval_width) == 0:
        raise ValueError("Too few elements for interval calculation. ")

    min_idx = np.argmin(interval_width)
    hdi_min = ary[min_idx]
    hdi_max = ary[min_idx + interval_idx_inc]

    if circular:
        hdi_min = hdi_min + mean
        hdi_max = hdi_max + mean
        hdi_min = np.arctan2(np.sin(hdi_min), np.cos(hdi_min))
        hdi_max = np.arctan2(np.sin(hdi_max), np.cos(hdi_max))

    hdi_interval = np.array([hdi_min, hdi_max])

    return hdi_interval


def _hdi_multimodal(ary, hdi_prob, skipna, max_modes):
    """Compute HDI if the distribution is multimodal."""
    ary = ary.flatten()
    if skipna:
        ary = ary[~np.isnan(ary)]

    if ary.dtype.kind == "f":
        bins, density = _kde(ary)
        lower, upper = bins[0], bins[-1]
        range_x = upper - lower
        dx = range_x / len(density)
    else:
        bins = _get_bins(ary)
        _, density, _ = _histogram(ary, bins=bins)
        dx = np.diff(bins)[0]

    density *= dx

    idx = np.argsort(-density)
    intervals = bins[idx][density[idx].cumsum() <= hdi_prob]
    intervals.sort()

    intervals_splitted = np.split(intervals, np.where(np.diff(intervals) >= dx * 1.1)[0] + 1)

    hdi_intervals = np.full((max_modes, 2), np.nan)
    for i, interval in enumerate(intervals_splitted):
        if i == max_modes:
            warnings.warn(
                f"found more modes than {max_modes}, returning only the first {max_modes} modes"
            )
            break
        if interval.size == 0:
            hdi_intervals[i] = np.asarray([bins[0], bins[0]])
        else:
            hdi_intervals[i] = np.asarray([interval[0], interval[-1]])

    return np.array(hdi_intervals)


def loo(data, pointwise=None, var_name=None, reff=None, scale=None):
    """Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).

    Estimates the expected log pointwise predictive density (elpd) using Pareto-smoothed
    importance sampling leave-one-out cross-validation (PSIS-LOO-CV). Also calculates LOO's
    standard error and the effective number of parameters. Read more theory here
    https://arxiv.org/abs/1507.04544 and here https://arxiv.org/abs/1507.02646

    Parameters
    ----------
    data: obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of
        :func:`arviz.convert_to_dataset` for details.
    pointwise: bool, optional
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``stats.ic_pointwise`` rcParam.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff: float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    scale: str
        Output scale for loo. Available options are:

        - ``log`` : (default) log-score
        - ``negative_log`` : -1 * log-score
        - ``deviance`` : -2 * log-score

        A higher log-score (or a lower deviance or negative log_score) indicates a model with
        better predictive accuracy.

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
    elpd: approximated expected log pointwise predictive density (elpd)
    se: standard error of the elpd
    p_loo: effective number of parameters
    shape_warn: bool
        True if the estimated shape parameter of
        Pareto distribution is greater than 0.7 for one or more samples
    loo_i: array of pointwise predictive accuracy, only if pointwise True
    pareto_k: array of Pareto shape values, only if pointwise True
    scale: scale of the elpd

        The returned object has a custom print method that overrides pd.Series method.

    See Also
    --------
    compare : Compare models based on PSIS-LOO loo or WAIC waic cross-validation.
    waic : Compute the widely applicable information criterion.
    plot_compare : Summary plot for model comparison.
    plot_elpd : Plot pointwise elpd differences between two or more models.
    plot_khat : Plot Pareto tail indices for diagnosing convergence.

    Examples
    --------
    Calculate LOO of a model:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("centered_eight")
           ...: az.loo(data)

    Calculate LOO of a model and return the pointwise values:

    .. ipython::

        In [2]: data_loo = az.loo(data, pointwise=True)
           ...: data_loo.loo_i
    """
    inference_data = convert_to_inference_data(data)
    log_likelihood = _get_log_likelihood(inference_data, var_name=var_name)
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))
    shape = log_likelihood.shape
    n_samples = shape[-1]
    n_data_points = np.prod(shape[:-1])
    scale = rcParams["stats.ic_scale"] if scale is None else scale.lower()

    if scale == "deviance":
        scale_value = -2
    elif scale == "log":
        scale_value = 1
    elif scale == "negative_log":
        scale_value = -1
    else:
        raise TypeError('Valid scale values are "deviance", "log", "negative_log"')

    if reff is None:
        if not hasattr(inference_data, "posterior"):
            raise TypeError("Must be able to extract a posterior group from data.")
        posterior = inference_data.posterior
        n_chains = len(posterior.chain)
        if n_chains == 1:
            reff = 1.0
        else:
            ess_p = ess(posterior, method="mean")
            # this mean is over all data variables
            reff = (
                np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples
            )

    log_weights, pareto_shape = psislw(-log_likelihood, reff)
    log_weights += log_likelihood

    warn_mg = False
    if np.any(pareto_shape > 0.7):
        warnings.warn(
            "Estimated shape parameter of Pareto distribution is greater than 0.7 for "
            "one or more samples. You should consider using a more robust model, this is because "
            "importance sampling is less likely to work well if the marginal posterior and "
            "LOO posterior are very different. This is more likely to happen with a non-robust "
            "model and highly influential observations."
        )
        warn_mg = True

    ufunc_kwargs = {"n_dims": 1, "ravel": False}
    kwargs = {"input_core_dims": [["__sample__"]]}
    loo_lppd_i = scale_value * _wrap_xarray_ufunc(
        _logsumexp, log_weights, ufunc_kwargs=ufunc_kwargs, **kwargs
    )
    loo_lppd = loo_lppd_i.values.sum()
    loo_lppd_se = (n_data_points * np.var(loo_lppd_i.values)) ** 0.5

    lppd = np.sum(
        _wrap_xarray_ufunc(
            _logsumexp,
            log_likelihood,
            func_kwargs={"b_inv": n_samples},
            ufunc_kwargs=ufunc_kwargs,
            **kwargs,
        ).values
    )
    p_loo = lppd - loo_lppd / scale_value

    if not pointwise:
        return ELPDData(
            data=[loo_lppd, loo_lppd_se, p_loo, n_samples, n_data_points, warn_mg, scale],
            index=["elpd_loo", "se", "p_loo", "n_samples", "n_data_points", "warning", "scale"],
        )
    if np.equal(loo_lppd, loo_lppd_i).all():  # pylint: disable=no-member
        warnings.warn(
            "The point-wise LOO is the same with the sum LOO, please double check "
            "the Observed RV in your model to make sure it returns element-wise logp."
        )
    return ELPDData(
        data=[
            loo_lppd,
            loo_lppd_se,
            p_loo,
            n_samples,
            n_data_points,
            warn_mg,
            loo_lppd_i.rename("loo_i"),
            pareto_shape,
            scale,
        ],
        index=[
            "elpd_loo",
            "se",
            "p_loo",
            "n_samples",
            "n_data_points",
            "warning",
            "loo_i",
            "pareto_k",
            "scale",
        ],
    )


def psislw(log_weights, reff=1.0):
    """
    Pareto smoothed importance sampling (PSIS).

    Notes
    -----
    If the ``log_weights`` input is an :class:`~xarray.DataArray` with a dimension
    named ``__sample__`` (recommended) ``psislw`` will interpret this dimension as samples,
    and all other dimensions as dimensions of the observed data, looping over them to
    calculate the psislw of each observation. If no ``__sample__`` dimension is present or
    the input is a numpy array, the last dimension will be interpreted as ``__sample__``.

    Parameters
    ----------
    log_weights: array
        Array of size (n_observations, n_samples)
    reff: float
        relative MCMC efficiency, ``ess / n``

    Returns
    -------
    lw_out: array
        Smoothed log weights
    kss: array
        Pareto tail indices

    References
    ----------
    * Vehtari et al. (2015) see https://arxiv.org/abs/1507.02646

    See Also
    --------
    loo : Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).

    Examples
    --------
    Get Pareto smoothed importance sampling (PSIS) log weights:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("non_centered_eight")
           ...: log_likelihood = data.log_likelihood["obs"].stack(
           ...:     __sample__=["chain", "draw"]
           ...: )
           ...: az.psislw(-log_likelihood, reff=0.8)

    """
    if hasattr(log_weights, "__sample__"):
        n_samples = len(log_weights.__sample__)
        shape = [
            size for size, dim in zip(log_weights.shape, log_weights.dims) if dim != "__sample__"
        ]
    else:
        n_samples = log_weights.shape[-1]
        shape = log_weights.shape[:-1]
    # precalculate constants
    cutoff_ind = -int(np.ceil(min(n_samples / 5.0, 3 * (n_samples / reff) ** 0.5))) - 1
    cutoffmin = np.log(np.finfo(float).tiny)  # pylint: disable=no-member, assignment-from-no-return

    # create output array with proper dimensions
    out = np.empty_like(log_weights), np.empty(shape)

    # define kwargs
    func_kwargs = {"cutoff_ind": cutoff_ind, "cutoffmin": cutoffmin, "out": out}
    ufunc_kwargs = {"n_dims": 1, "n_output": 2, "ravel": False, "check_shape": False}
    kwargs = {"input_core_dims": [["__sample__"]], "output_core_dims": [["__sample__"], []]}
    log_weights, pareto_shape = _wrap_xarray_ufunc(
        _psislw,
        log_weights,
        ufunc_kwargs=ufunc_kwargs,
        func_kwargs=func_kwargs,
        **kwargs,
    )
    if isinstance(log_weights, xr.DataArray):
        log_weights = log_weights.rename("log_weights")
    if isinstance(pareto_shape, xr.DataArray):
        pareto_shape = pareto_shape.rename("pareto_shape")
    return log_weights, pareto_shape


def _psislw(log_weights, cutoff_ind, cutoffmin):
    """
    Pareto smoothed importance sampling (PSIS) for a 1D vector.

    Parameters
    ----------
    log_weights: array
        Array of length n_observations
    cutoff_ind: int
    cutoffmin: float
    k_min: float

    Returns
    -------
    lw_out: array
        Smoothed log weights
    kss: float
        Pareto tail index
    """
    x = np.asarray(log_weights)

    # improve numerical accuracy
    x -= np.max(x)
    # sort the array
    x_sort_ind = np.argsort(x)
    # divide log weights into body and right tail
    xcutoff = max(x[x_sort_ind[cutoff_ind]], cutoffmin)

    expxcutoff = np.exp(xcutoff)
    (tailinds,) = np.where(x > xcutoff)  # pylint: disable=unbalanced-tuple-unpacking
    x_tail = x[tailinds]
    tail_len = len(x_tail)
    if tail_len <= 4:
        # not enough tail samples for gpdfit
        k = np.inf
    else:
        # order of tail samples
        x_tail_si = np.argsort(x_tail)
        # fit generalized Pareto distribution to the right tail samples
        x_tail = np.exp(x_tail) - expxcutoff
        k, sigma = _gpdfit(x_tail[x_tail_si])

        if np.isfinite(k):
            # no smoothing if GPD fit failed
            # compute ordered statistic for the fit
            sti = np.arange(0.5, tail_len) / tail_len
            smoothed_tail = _gpinv(sti, k, sigma)
            smoothed_tail = np.log(  # pylint: disable=assignment-from-no-return
                smoothed_tail + expxcutoff
            )
            # place the smoothed tail into the output array
            x[tailinds[x_tail_si]] = smoothed_tail
            # truncate smoothed values to the largest raw weight 0
            x[x > 0] = 0
    # renormalize weights
    x -= _logsumexp(x)

    return x, k


def _gpdfit(ary):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD).

    Empirical Bayes estimate for the parameters of the generalized Pareto
    distribution given the data.

    Parameters
    ----------
    ary: array
        sorted 1D data array

    Returns
    -------
    k: float
        estimated shape parameter
    sigma: float
        estimated scale parameter
    """
    prior_bs = 3
    prior_k = 10
    n = len(ary)
    m_est = 30 + int(n**0.5)

    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1 / ary[-1]

    k_ary = np.log1p(-b_ary[:, None] * ary).mean(axis=1)  # pylint: disable=no-member
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)

    # remove negligible weights
    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]
    # normalise weights
    weights /= weights.sum()

    # posterior mean for b
    b_post = np.sum(b_ary * weights)
    # estimate for k
    k_post = np.log1p(-b_post * ary).mean()  # pylint: disable=invalid-unary-operand-type,no-member
    # add prior for k_post
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)

    return k_post, sigma


def _gpinv(probs, kappa, sigma):
    """Inverse Generalized Pareto distribution function."""
    # pylint: disable=unsupported-assignment-operation, invalid-unary-operand-type
    x = np.full_like(probs, np.nan)
    if sigma <= 0:
        return x
    ok = (probs > 0) & (probs < 1)
    if np.all(ok):
        if np.abs(kappa) < np.finfo(float).eps:
            x = -np.log1p(-probs)
        else:
            x = np.expm1(-kappa * np.log1p(-probs)) / kappa
        x *= sigma
    else:
        if np.abs(kappa) < np.finfo(float).eps:
            x[ok] = -np.log1p(-probs[ok])
        else:
            x[ok] = np.expm1(-kappa * np.log1p(-probs[ok])) / kappa
        x *= sigma
        x[probs == 0] = 0
        x[probs == 1] = np.inf if kappa >= 0 else -sigma / kappa
    return x


def r2_samples(y_true, y_pred):
    """R² samples for Bayesian regression models. Only valid for linear models.

    Parameters
    ----------
    y_true: array-like of shape = (n_outputs,)
        Ground truth (correct) target values.
    y_pred: array-like of shape = (n_posterior_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    Pandas Series with the following indices:
    Bayesian R² samples.

    See Also
    --------
    plot_lm : Posterior predictive and mean plots for regression-like data.

    Examples
    --------
    Calculate R² samples for Bayesian regression models :

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data('regression1d')
           ...: y_true = data.observed_data["y"].values
           ...: y_pred = data.posterior_predictive.stack(sample=("chain", "draw"))["y"].values.T
           ...: az.r2_samples(y_true, y_pred)

    """
    _numba_flag = Numba.numba_flag
    if y_pred.ndim == 1:
        var_y_est = _numba_var(svar, np.var, y_pred)
        var_e = _numba_var(svar, np.var, (y_true - y_pred))
    else:
        var_y_est = _numba_var(svar, np.var, y_pred, axis=1)
        var_e = _numba_var(svar, np.var, (y_true - y_pred), axis=1)
    r_squared = var_y_est / (var_y_est + var_e)

    return r_squared


def r2_score(y_true, y_pred):
    """R² for Bayesian regression models. Only valid for linear models.

    Parameters
    ----------
    y_true: array-like of shape = (n_outputs,)
        Ground truth (correct) target values.
    y_pred: array-like of shape = (n_posterior_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    Pandas Series with the following indices:
    r2: Bayesian R²
    r2_std: standard deviation of the Bayesian R².

    See Also
    --------
    plot_lm : Posterior predictive and mean plots for regression-like data.

    Examples
    --------
    Calculate R² for Bayesian regression models :

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data('regression1d')
           ...: y_true = data.observed_data["y"].values
           ...: y_pred = data.posterior_predictive.stack(sample=("chain", "draw"))["y"].values.T
           ...: az.r2_score(y_true, y_pred)

    """
    r_squared = r2_samples(y_true=y_true, y_pred=y_pred)
    return pd.Series([np.mean(r_squared), np.std(r_squared)], index=["r2", "r2_std"])


def summary(
    data,
    var_names: Optional[List[str]] = None,
    filter_vars=None,
    group=None,
    fmt: "Literal['wide', 'long', 'xarray']" = "wide",
    kind: "Literal['all', 'stats', 'diagnostics']" = "all",
    round_to=None,
    circ_var_names=None,
    stat_focus="mean",
    stat_funcs=None,
    extend=True,
    hdi_prob=None,
    skipna=False,
    labeller=None,
    coords=None,
    index_origin=None,
    order=None,
) -> Union[pd.DataFrame, xr.Dataset]:
    """Create a data frame with summary statistics.

    Parameters
    ----------
    data: obj
        Any object that can be converted to an :class:`arviz.InferenceData` object
        Refer to documentation of :func:`arviz.convert_to_dataset` for details
    var_names: list
        Names of variables to include in summary. Prefix the variables by ``~`` when you
        want to exclude them from the summary: `["~beta"]` instead of `["beta"]` (see
        examples below).
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
        interpret var_names as substrings of the real variables names. If "regex",
        interpret var_names as regular expressions on the real variables names. A la
        ``pandas.filter``.
    coords: Dict[str, List[Any]], optional
        Coordinate subset for which to calculate the summary.
    group: str
        Select a group for summary. Defaults to "posterior", "prior" or first group
        in that order, depending what groups exists.
    fmt: {'wide', 'long', 'xarray'}
        Return format is either pandas.DataFrame {'wide', 'long'} or xarray.Dataset {'xarray'}.
    kind: {'all', 'stats', 'diagnostics'}
        Whether to include the `stats`: `mean`, `sd`, `hdi_3%`, `hdi_97%`, or the `diagnostics`:
        `mcse_mean`, `mcse_sd`, `ess_bulk`, `ess_tail`, and `r_hat`. Default to include `all` of
        them.
    round_to: int
        Number of decimals used to round results. Defaults to 2. Use "none" to return raw numbers.
    circ_var_names: list
        A list of circular variables to compute circular stats for
    stat_focus : str, default "mean"
        Select the focus for summary.
    stat_funcs: dict
        A list of functions or a dict of functions with function names as keys used to calculate
        statistics. By default, the mean, standard deviation, simulation standard error, and
        highest posterior density intervals are included.

        The functions will be given one argument, the samples for a variable as an nD array,
        The functions should be in the style of a ufunc and return a single number. For example,
        :func:`numpy.mean`, or ``scipy.stats.var`` would both work.
    extend: boolean
        If True, use the statistics returned by ``stat_funcs`` in addition to, rather than in place
        of, the default statistics. This is only meaningful when ``stat_funcs`` is not None.
    hdi_prob: float, optional
        Highest density interval to compute. Defaults to 0.94. This is only meaningful when
        ``stat_funcs`` is None.
    skipna: bool
        If true ignores nan values when computing the summary statistics, it does not affect the
        behaviour of the functions passed to ``stat_funcs``. Defaults to false.
    labeller : labeller instance, optional
        Class providing the method `make_label_flat` to generate the labels in the plot titles.
        For more details on ``labeller`` usage see :ref:`label_guide`
    credible_interval: float, optional
        deprecated: Please see hdi_prob
    order
        deprecated: order is now ignored.
    index_origin
        deprecated: index_origin is now ignored, modify the coordinate values to change the
        value used in summary.

    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        Return type dicated by `fmt` argument.

        Return value will contain summary statistics for each variable. Default statistics depend on
        the value of ``stat_focus``:

        ``stat_focus="mean"``: `mean`, `sd`, `hdi_3%`, `hdi_97%`, `mcse_mean`, `mcse_sd`,
        `ess_bulk`, `ess_tail`, and `r_hat`

        ``stat_focus="median"``: `median`, `mad`, `eti_3%`, `eti_97%`, `mcse_median`, `ess_median`,
        `ess_tail`, and `r_hat`

        `r_hat` is only computed for traces with 2 or more chains.

    See Also
    --------
    waic : Compute the widely applicable information criterion.
    loo : Compute Pareto-smoothed importance sampling leave-one-out
          cross-validation (PSIS-LOO-CV).
    ess : Calculate estimate of the effective sample size (ess).
    rhat : Compute estimate of rank normalized splitR-hat for a set of traces.
    mcse : Calculate Markov Chain Standard Error statistic.

    Examples
    --------
    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("centered_eight")
           ...: az.summary(data, var_names=["mu", "tau"])

    You can use ``filter_vars`` to select variables without having to specify all the exact
    names. Use ``filter_vars="like"`` to select based on partial naming:

    .. ipython::

        In [1]: az.summary(data, var_names=["the"], filter_vars="like")

    Use ``filter_vars="regex"`` to select based on regular expressions, and prefix the variables
    you want to exclude by ``~``. Here, we exclude from the summary all the variables
    starting with the letter t:

    .. ipython::

        In [1]: az.summary(data, var_names=["~^t"], filter_vars="regex")

    Other statistics can be calculated by passing a list of functions
    or a dictionary with key, function pairs.

    .. ipython::

        In [1]: import numpy as np
           ...: def median_sd(x):
           ...:     median = np.percentile(x, 50)
           ...:     sd = np.sqrt(np.mean((x-median)**2))
           ...:     return sd
           ...:
           ...: func_dict = {
           ...:     "std": np.std,
           ...:     "median_std": median_sd,
           ...:     "5%": lambda x: np.percentile(x, 5),
           ...:     "median": lambda x: np.percentile(x, 50),
           ...:     "95%": lambda x: np.percentile(x, 95),
           ...: }
           ...: az.summary(
           ...:     data,
           ...:     var_names=["mu", "tau"],
           ...:     stat_funcs=func_dict,
           ...:     extend=False
           ...: )

    Use ``stat_focus`` to change the focus of summary statistics obatined to median:

    .. ipython::

        In [1]: az.summary(data, stat_focus="median")

    """
    _log.cache = []

    if coords is None:
        coords = {}

    if index_origin is not None:
        warnings.warn(
            "index_origin has been deprecated. summary now shows coordinate values, "
            "to change the label shown, modify the coordinate values before calling summary",
            DeprecationWarning,
        )
        index_origin = rcParams["data.index_origin"]
    if labeller is None:
        labeller = BaseLabeller()
    if hdi_prob is None:
        hdi_prob = rcParams["stats.hdi_prob"]
    elif not 1 >= hdi_prob > 0:
        raise ValueError("The value of hdi_prob should be in the interval (0, 1]")

    if isinstance(data, InferenceData):
        if group is None:
            if not data.groups():
                raise TypeError("InferenceData does not contain any groups")
            if "posterior" in data:
                dataset = data["posterior"]
            elif "prior" in data:
                dataset = data["prior"]
            else:
                warnings.warn(f"Selecting first found group: {data.groups()[0]}")
                dataset = data[data.groups()[0]]
        elif group in data.groups():
            dataset = data[group]
        else:
            raise TypeError(f"InferenceData does not contain group: {group}")
    else:
        dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset, filter_vars)
    dataset = dataset if var_names is None else dataset[var_names]
    dataset = get_coords(dataset, coords)

    fmt_group = ("wide", "long", "xarray")
    if not isinstance(fmt, str) or (fmt.lower() not in fmt_group):
        raise TypeError(f"Invalid format: '{fmt}'. Formatting options are: {fmt_group}")

    kind_group = ("all", "stats", "diagnostics")
    if not isinstance(kind, str) or kind not in kind_group:
        raise TypeError(f"Invalid kind: '{kind}'. Kind options are: {kind_group}")

    focus_group = ("mean", "median")
    if not isinstance(stat_focus, str) or (stat_focus not in focus_group):
        raise TypeError(f"Invalid format: '{stat_focus}'. Focus options are: {focus_group}")

    if stat_focus != "mean" and circ_var_names is not None:
        raise TypeError(f"Invalid format: Circular stats not supported for '{stat_focus}'")

    if order is not None:
        warnings.warn(
            "order has been deprecated. summary now shows coordinate values.", DeprecationWarning
        )

    alpha = 1 - hdi_prob

    extra_metrics = []
    extra_metric_names = []

    if stat_funcs is not None:
        if isinstance(stat_funcs, dict):
            for stat_func_name, stat_func in stat_funcs.items():
                extra_metrics.append(
                    xr.apply_ufunc(
                        _make_ufunc(stat_func), dataset, input_core_dims=(("chain", "draw"),)
                    )
                )
                extra_metric_names.append(stat_func_name)
        else:
            for stat_func in stat_funcs:
                extra_metrics.append(
                    xr.apply_ufunc(
                        _make_ufunc(stat_func), dataset, input_core_dims=(("chain", "draw"),)
                    )
                )
                extra_metric_names.append(stat_func.__name__)

    metrics: List[xr.Dataset] = []
    metric_names: List[str] = []
    if extend and kind in ["all", "stats"]:
        if stat_focus == "mean":
            mean = dataset.mean(dim=("chain", "draw"), skipna=skipna)

            sd = dataset.std(dim=("chain", "draw"), ddof=1, skipna=skipna)

            hdi_post = hdi(dataset, hdi_prob=hdi_prob, multimodal=False, skipna=skipna)
            hdi_lower = hdi_post.sel(hdi="lower", drop=True)
            hdi_higher = hdi_post.sel(hdi="higher", drop=True)
            metrics.extend((mean, sd, hdi_lower, hdi_higher))
            metric_names.extend(
                ("mean", "sd", f"hdi_{100 * alpha / 2:g}%", f"hdi_{100 * (1 - alpha / 2):g}%")
            )
        elif stat_focus == "median":
            median = dataset.median(dim=("chain", "draw"), skipna=skipna)

            mad = stats.median_abs_deviation(dataset, dims=("chain", "draw"))
            eti_post = dataset.quantile(
                (alpha / 2, 1 - alpha / 2), dim=("chain", "draw"), skipna=skipna
            )
            eti_lower = eti_post.isel(quantile=0, drop=True)
            eti_higher = eti_post.isel(quantile=1, drop=True)
            metrics.extend((median, mad, eti_lower, eti_higher))
            metric_names.extend(
                ("median", "mad", f"eti_{100 * alpha / 2:g}%", f"eti_{100 * (1 - alpha / 2):g}%")
            )

    if circ_var_names:
        nan_policy = "omit" if skipna else "propagate"
        circ_mean = stats.circmean(
            dataset, dims=["chain", "draw"], high=np.pi, low=-np.pi, nan_policy=nan_policy
        )
        _numba_flag = Numba.numba_flag
        if _numba_flag:
            circ_sd = xr.apply_ufunc(
                _make_ufunc(_circular_standard_deviation),
                dataset,
                kwargs=dict(high=np.pi, low=-np.pi, skipna=skipna),
                input_core_dims=(("chain", "draw"),),
            )
        else:
            circ_sd = stats.circstd(
                dataset, dims=["chain", "draw"], high=np.pi, low=-np.pi, nan_policy=nan_policy
            )
        circ_mcse = xr.apply_ufunc(
            _make_ufunc(_mc_error),
            dataset,
            kwargs=dict(circular=True),
            input_core_dims=(("chain", "draw"),),
        )

        circ_hdi = hdi(dataset, hdi_prob=hdi_prob, circular=True, skipna=skipna)
        circ_hdi_lower = circ_hdi.sel(hdi="lower", drop=True)
        circ_hdi_higher = circ_hdi.sel(hdi="higher", drop=True)

    if kind in ["all", "diagnostics"] and extend:
        diagnostics_names: Tuple[str, ...]
        if stat_focus == "mean":
            diagnostics = xr.apply_ufunc(
                _make_ufunc(_multichain_statistics, n_output=5, ravel=False),
                dataset,
                input_core_dims=(("chain", "draw"),),
                output_core_dims=tuple([] for _ in range(5)),
            )
            diagnostics_names = (
                "mcse_mean",
                "mcse_sd",
                "ess_bulk",
                "ess_tail",
                "r_hat",
            )

        elif stat_focus == "median":
            diagnostics = xr.apply_ufunc(
                _make_ufunc(_multichain_statistics, n_output=4, ravel=False),
                dataset,
                kwargs=dict(focus="median"),
                input_core_dims=(("chain", "draw"),),
                output_core_dims=tuple([] for _ in range(4)),
            )
            diagnostics_names = (
                "mcse_median",
                "ess_median",
                "ess_tail",
                "r_hat",
            )
        metrics.extend(diagnostics)
        metric_names.extend(diagnostics_names)

    if circ_var_names and kind != "diagnostics" and stat_focus == "mean":
        for metric, circ_stat in zip(
            # Replace only the first 5 statistics for their circular equivalent
            metrics[:5],
            (circ_mean, circ_sd, circ_hdi_lower, circ_hdi_higher, circ_mcse),
        ):
            for circ_var in circ_var_names:
                metric[circ_var] = circ_stat[circ_var]

    metrics.extend(extra_metrics)
    metric_names.extend(extra_metric_names)
    joined = (
        xr.concat(metrics, dim="metric").assign_coords(metric=metric_names).reset_coords(drop=True)
    )
    n_metrics = len(metric_names)
    n_vars = np.sum([joined[var].size // n_metrics for var in joined.data_vars])

    if fmt.lower() == "wide":
        summary_df = pd.DataFrame(
            (np.full((cast(int, n_vars), n_metrics), np.nan)), columns=metric_names
        )
        indices = []
        for i, (var_name, sel, isel, values) in enumerate(
            xarray_var_iter(joined, skip_dims={"metric"})
        ):
            summary_df.iloc[i] = values
            indices.append(labeller.make_label_flat(var_name, sel, isel))
        summary_df.index = indices
    elif fmt.lower() == "long":
        df = joined.to_dataframe().reset_index().set_index("metric")
        df.index = list(df.index)
        summary_df = df
    else:
        # format is 'xarray'
        summary_df = joined
    if (round_to is not None) and (round_to not in ("None", "none")):
        summary_df = summary_df.round(round_to)
    elif round_to not in ("None", "none") and (fmt.lower() in ("long", "wide")):
        # Don't round xarray object by default (even with "none")
        decimals = {
            col: 3 if col not in {"ess_bulk", "ess_tail", "r_hat"} else 2 if col == "r_hat" else 0
            for col in summary_df.columns
        }
        summary_df = summary_df.round(decimals)

    return summary_df


def waic(data, pointwise=None, var_name=None, scale=None, dask_kwargs=None):
    """Compute the widely applicable information criterion.

    Estimates the expected log pointwise predictive density (elpd) using WAIC. Also calculates the
    WAIC's standard error and the effective number of parameters.
    Read more theory here https://arxiv.org/abs/1507.04544 and here https://arxiv.org/abs/1004.2316

    Parameters
    ----------
    data: obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_inference_data` for details.
    pointwise: bool
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``stats.ic_pointwise`` rcParam.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for waic computation.
    scale: str
        Output scale for WAIC. Available options are:

        - `log` : (default) log-score
        - `negative_log` : -1 * log-score
        - `deviance` : -2 * log-score

        A higher log-score (or a lower deviance or negative log_score) indicates a model with
        better predictive accuracy.
    dask_kwargs : dict, optional
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
    elpd_waic: approximated expected log pointwise predictive density (elpd)
    se: standard error of the elpd
    p_waic: effective number parameters
    var_warn: bool
        True if posterior variance of the log predictive densities exceeds 0.4
    waic_i: :class:`~xarray.DataArray` with the pointwise predictive accuracy,
            only if pointwise=True
    scale: scale of the elpd

        The returned object has a custom print method that overrides pd.Series method.

    See Also
    --------
    loo : Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).
    compare : Compare models based on PSIS-LOO-CV or WAIC.
    plot_compare : Summary plot for model comparison.

    Examples
    --------
    Calculate WAIC of a model:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("centered_eight")
           ...: az.waic(data)

    Calculate WAIC of a model and return the pointwise values:

    .. ipython::

        In [2]: data_waic = az.waic(data, pointwise=True)
           ...: data_waic.waic_i
    """
    inference_data = convert_to_inference_data(data)
    log_likelihood = _get_log_likelihood(inference_data, var_name=var_name)
    scale = rcParams["stats.ic_scale"] if scale is None else scale.lower()
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    if scale == "deviance":
        scale_value = -2
    elif scale == "log":
        scale_value = 1
    elif scale == "negative_log":
        scale_value = -1
    else:
        raise TypeError('Valid scale values are "deviance", "log", "negative_log"')

    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))
    shape = log_likelihood.shape
    n_samples = shape[-1]
    n_data_points = np.prod(shape[:-1])

    ufunc_kwargs = {"n_dims": 1, "ravel": False}
    kwargs = {"input_core_dims": [["__sample__"]]}
    lppd_i = _wrap_xarray_ufunc(
        _logsumexp,
        log_likelihood,
        func_kwargs={"b_inv": n_samples},
        ufunc_kwargs=ufunc_kwargs,
        dask_kwargs=dask_kwargs,
        **kwargs,
    )

    vars_lpd = log_likelihood.var(dim="__sample__")
    warn_mg = False
    if np.any(vars_lpd > 0.4):
        warnings.warn(
            (
                "For one or more samples the posterior variance of the log predictive "
                "densities exceeds 0.4. This could be indication of WAIC starting to fail. \n"
                "See http://arxiv.org/abs/1507.04544 for details"
            )
        )
        warn_mg = True

    waic_i = scale_value * (lppd_i - vars_lpd)
    waic_se = (n_data_points * np.var(waic_i.values)) ** 0.5
    waic_sum = np.sum(waic_i.values)
    p_waic = np.sum(vars_lpd.values)

    if not pointwise:
        return ELPDData(
            data=[waic_sum, waic_se, p_waic, n_samples, n_data_points, warn_mg, scale],
            index=[
                "waic",
                "se",
                "p_waic",
                "n_samples",
                "n_data_points",
                "warning",
                "scale",
            ],
        )
    if np.equal(waic_sum, waic_i).all():  # pylint: disable=no-member
        warnings.warn(
            """The point-wise WAIC is the same with the sum WAIC, please double check
            the Observed RV in your model to make sure it returns element-wise logp.
            """
        )
    return ELPDData(
        data=[
            waic_sum,
            waic_se,
            p_waic,
            n_samples,
            n_data_points,
            warn_mg,
            waic_i.rename("waic_i"),
            scale,
        ],
        index=[
            "elpd_waic",
            "se",
            "p_waic",
            "n_samples",
            "n_data_points",
            "warning",
            "waic_i",
            "scale",
        ],
    )


def loo_pit(idata=None, *, y=None, y_hat=None, log_weights=None):
    """Compute leave one out (PSIS-LOO) probability integral transform (PIT) values.

    Parameters
    ----------
    idata: InferenceData
        :class:`arviz.InferenceData` object.
    y: array, DataArray or str
        Observed data. If str, ``idata`` must be present and contain the observed data group
    y_hat: array, DataArray or str
        Posterior predictive samples for ``y``. It must have the same shape as y plus an
        extra dimension at the end of size n_samples (chains and draws stacked). If str or
        None, ``idata`` must contain the posterior predictive group. If None, y_hat is taken
        equal to y, thus, y must be str too.
    log_weights: array or DataArray
        Smoothed log_weights. It must have the same shape as ``y_hat``
    dask_kwargs : dict, optional
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    loo_pit: array or DataArray
        Value of the LOO-PIT at each observed data point.

    See Also
    --------
    plot_loo_pit : Plot Leave-One-Out probability integral transformation (PIT) predictive checks.
    loo : Compute Pareto-smoothed importance sampling leave-one-out
          cross-validation (PSIS-LOO-CV).
    plot_elpd : Plot pointwise elpd differences between two or more models.
    plot_khat : Plot Pareto tail indices for diagnosing convergence.

    Examples
    --------
    Calculate LOO-PIT values using as test quantity the observed values themselves.

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data("centered_eight")
           ...: az.loo_pit(idata=data, y="obs")

    Calculate LOO-PIT values using as test quantity the square of the difference between
    each observation and `mu`. Both ``y`` and ``y_hat`` inputs will be array-like,
    but ``idata`` will still be passed in order to calculate the ``log_weights`` from
    there.

    .. ipython::

        In [1]: T = data.observed_data.obs - data.posterior.mu.median(dim=("chain", "draw"))
           ...: T_hat = data.posterior_predictive.obs - data.posterior.mu
           ...: T_hat = T_hat.stack(__sample__=("chain", "draw"))
           ...: az.loo_pit(idata=data, y=T**2, y_hat=T_hat**2)

    """
    y_str = ""
    if idata is not None and not isinstance(idata, InferenceData):
        raise ValueError("idata must be of type InferenceData or None")

    if idata is None:
        if not all(isinstance(arg, (np.ndarray, xr.DataArray)) for arg in (y, y_hat, log_weights)):
            raise ValueError(
                "all 3 y, y_hat and log_weights must be array or DataArray when idata is None "
                f"but they are of types {[type(arg) for arg in (y, y_hat, log_weights)]}"
            )

    else:
        if y_hat is None and isinstance(y, str):
            y_hat = y
        elif y_hat is None:
            raise ValueError("y_hat cannot be None if y is not a str")
        if isinstance(y, str):
            y_str = y
            y = idata.observed_data[y].values
        elif not isinstance(y, (np.ndarray, xr.DataArray)):
            raise ValueError(f"y must be of types array, DataArray or str, not {type(y)}")
        if isinstance(y_hat, str):
            y_hat = idata.posterior_predictive[y_hat].stack(__sample__=("chain", "draw")).values
        elif not isinstance(y_hat, (np.ndarray, xr.DataArray)):
            raise ValueError(f"y_hat must be of types array, DataArray or str, not {type(y_hat)}")
        if log_weights is None:
            if y_str:
                try:
                    log_likelihood = _get_log_likelihood(idata, var_name=y_str)
                except TypeError:
                    log_likelihood = _get_log_likelihood(idata)
            else:
                log_likelihood = _get_log_likelihood(idata)
            log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))
            posterior = convert_to_dataset(idata, group="posterior")
            n_chains = len(posterior.chain)
            n_samples = len(log_likelihood.__sample__)
            ess_p = ess(posterior, method="mean")
            # this mean is over all data variables
            reff = (
                (np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples)
                if n_chains > 1
                else 1
            )
            log_weights = psislw(-log_likelihood, reff=reff)[0].values
        elif not isinstance(log_weights, (np.ndarray, xr.DataArray)):
            raise ValueError(
                f"log_weights must be None or of types array or DataArray, not {type(log_weights)}"
            )

    if len(y.shape) + 1 != len(y_hat.shape):
        raise ValueError(
            f"y_hat must have 1 more dimension than y, but y_hat has {len(y_hat.shape)} dims and "
            f"y has {len(y.shape)} dims"
        )

    if y.shape != y_hat.shape[:-1]:
        raise ValueError(
            f"y has shape: {y.shape} which should be equal to y_hat shape (omitting the last "
            f"dimension): {y_hat.shape}"
        )

    if y_hat.shape != log_weights.shape:
        raise ValueError(
            "y_hat and log_weights must have the same shape but have shapes "
            f"{y_hat.shape,} and {log_weights.shape}"
        )

    kwargs = {
        "input_core_dims": [[], ["__sample__"], ["__sample__"]],
        "output_core_dims": [[]],
        "join": "left",
    }
    ufunc_kwargs = {"n_dims": 1}

    if y.dtype.kind == "i" or y_hat.dtype.kind == "i":
        y, y_hat = smooth_data(y, y_hat)

    return _wrap_xarray_ufunc(
        _loo_pit,
        y,
        y_hat,
        log_weights,
        ufunc_kwargs=ufunc_kwargs,
        **kwargs,
    )


def _loo_pit(y, y_hat, log_weights):
    """Compute LOO-PIT values."""
    sel = y_hat <= y
    if np.sum(sel) > 0:
        value = np.exp(_logsumexp(log_weights[sel]))
        return min(1, value)
    else:
        return 0


def apply_test_function(
    idata,
    func,
    group="both",
    var_names=None,
    pointwise=False,
    out_data_shape=None,
    out_pp_shape=None,
    out_name_data="T",
    out_name_pp=None,
    func_args=None,
    func_kwargs=None,
    ufunc_kwargs=None,
    wrap_data_kwargs=None,
    wrap_pp_kwargs=None,
    inplace=True,
    overwrite=None,
):
    """Apply a Bayesian test function to an InferenceData object.

    Parameters
    ----------
    idata: InferenceData
        :class:`arviz.InferenceData` object on which to apply the test function.
        This function will add new variables to the InferenceData object
        to store the result without modifying the existing ones.
    func: callable
        Callable that calculates the test function. It must have the following call signature
        ``func(y, theta, *args, **kwargs)`` (where ``y`` is the observed data or posterior
        predictive and ``theta`` the model parameters) even if not all the arguments are
        used.
    group: str, optional
        Group on which to apply the test function. Can be observed_data, posterior_predictive
        or both.
    var_names: dict group -> var_names, optional
        Mapping from group name to the variables to be passed to func. It can be a dict of
        strings or lists of strings. There is also the option of using ``both`` as key,
        in which case, the same variables are used in observed data and posterior predictive
        groups
    pointwise: bool, optional
        If True, apply the test function to each observation and sample, otherwise, apply
        test function to each sample.
    out_data_shape, out_pp_shape: tuple, optional
        Output shape of the test function applied to the observed/posterior predictive data.
        If None, the default depends on the value of pointwise.
    out_name_data, out_name_pp: str, optional
        Name of the variables to add to the observed_data and posterior_predictive datasets
        respectively. ``out_name_pp`` can be ``None``, in which case will be taken equal to
        ``out_name_data``.
    func_args: sequence, optional
        Passed as is to ``func``
    func_kwargs: mapping, optional
        Passed as is to ``func``
    wrap_data_kwargs, wrap_pp_kwargs: mapping, optional
        kwargs passed to :func:`~arviz.wrap_xarray_ufunc`. By default, some suitable input_core_dims
        are used.
    inplace: bool, optional
        If True, add the variables inplace, otherwise, return a copy of idata with the variables
        added.
    overwrite: bool, optional
        Overwrite data in case ``out_name_data`` or ``out_name_pp`` are already variables in
        dataset. If ``None`` it will be the opposite of inplace.

    Returns
    -------
    idata: InferenceData
        Output InferenceData object. If ``inplace=True``, it is the same input object modified
        inplace.

    See Also
    --------
    plot_bpv :  Plot Bayesian p-value for observed data and Posterior/Prior predictive.

    Notes
    -----
    This function is provided for convenience to wrap scalar or functions working on low
    dims to inference data object. It is not optimized to be faster nor as fast as vectorized
    computations.

    Examples
    --------
    Use ``apply_test_function`` to wrap ``numpy.min`` for illustration purposes. And plot the
    results.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> idata = az.load_arviz_data("centered_eight")
        >>> az.apply_test_function(idata, lambda y, theta: np.min(y))
        >>> T = idata.observed_data.T.item()
        >>> az.plot_posterior(idata, var_names=["T"], group="posterior_predictive", ref_val=T)

    """
    out = idata if inplace else deepcopy(idata)

    valid_groups = ("observed_data", "posterior_predictive", "both")
    if group not in valid_groups:
        raise ValueError(f"Invalid group argument. Must be one of {valid_groups} not {group}.")
    if overwrite is None:
        overwrite = not inplace

    if out_name_pp is None:
        out_name_pp = out_name_data

    if func_args is None:
        func_args = tuple()

    if func_kwargs is None:
        func_kwargs = {}

    if ufunc_kwargs is None:
        ufunc_kwargs = {}
    ufunc_kwargs.setdefault("check_shape", False)
    ufunc_kwargs.setdefault("ravel", False)

    if wrap_data_kwargs is None:
        wrap_data_kwargs = {}
    if wrap_pp_kwargs is None:
        wrap_pp_kwargs = {}
    if var_names is None:
        var_names = {}

    both_var_names = var_names.pop("both", None)
    var_names.setdefault("posterior", list(out.posterior.data_vars))

    in_posterior = out.posterior[var_names["posterior"]]
    if isinstance(in_posterior, xr.Dataset):
        in_posterior = in_posterior.to_array().squeeze()

    groups = ("posterior_predictive", "observed_data") if group == "both" else [group]
    for grp in groups:
        out_group_shape = out_data_shape if grp == "observed_data" else out_pp_shape
        out_name_group = out_name_data if grp == "observed_data" else out_name_pp
        wrap_group_kwargs = wrap_data_kwargs if grp == "observed_data" else wrap_pp_kwargs
        if not hasattr(out, grp):
            raise ValueError(f"InferenceData object must have {grp} group")
        if not overwrite and out_name_group in getattr(out, grp).data_vars:
            raise ValueError(
                f"Should overwrite: {out_name_group} variable present in group {grp},"
                " but overwrite is False"
            )
        var_names.setdefault(
            grp, list(getattr(out, grp).data_vars) if both_var_names is None else both_var_names
        )
        in_group = getattr(out, grp)[var_names[grp]]
        if isinstance(in_group, xr.Dataset):
            in_group = in_group.to_array(dim=f"{grp}_var").squeeze()

        if pointwise:
            out_group_shape = in_group.shape if out_group_shape is None else out_group_shape
        elif grp == "observed_data":
            out_group_shape = () if out_group_shape is None else out_group_shape
        elif grp == "posterior_predictive":
            out_group_shape = in_group.shape[:2] if out_group_shape is None else out_group_shape
        loop_dims = in_group.dims[: len(out_group_shape)]

        wrap_group_kwargs.setdefault(
            "input_core_dims",
            [
                [dim for dim in dataset.dims if dim not in loop_dims]
                for dataset in [in_group, in_posterior]
            ],
        )
        func_kwargs["out"] = np.empty(out_group_shape)

        out_group = getattr(out, grp)
        try:
            out_group[out_name_group] = _wrap_xarray_ufunc(
                func,
                in_group.values,
                in_posterior.values,
                func_args=func_args,
                func_kwargs=func_kwargs,
                ufunc_kwargs=ufunc_kwargs,
                **wrap_group_kwargs,
            )
        except IndexError:
            excluded_dims = set(
                wrap_group_kwargs["input_core_dims"][0] + wrap_group_kwargs["input_core_dims"][1]
            )
            out_group[out_name_group] = _wrap_xarray_ufunc(
                func,
                *xr.broadcast(in_group, in_posterior, exclude=excluded_dims),
                func_args=func_args,
                func_kwargs=func_kwargs,
                ufunc_kwargs=ufunc_kwargs,
                **wrap_group_kwargs,
            )
        setattr(out, grp, out_group)

    return out


def weight_predictions(idatas, weights=None):
    """
    Generate weighted posterior predictive samples from a list of InferenceData
    and a set of weights.

    Parameters
    ---------
    idatas : list[InferenceData]
        List of :class:`arviz.InferenceData` objects containing the groups `posterior_predictive`
        and `observed_data`. Observations should be the same for all InferenceData objects.
    weights : array-like, optional
        Individual weights for each model. Weights should be positive. If they do not sum up to 1,
        they will be normalized. Default, same weight for each model.
        Weights can be computed using many different methods including those in
        :func:`arviz.compare`.

    Returns
    -------
    idata: InferenceData
        Output InferenceData object with the groups `posterior_predictive` and `observed_data`.

    See Also
    --------
    compare :  Compare models based on PSIS-LOO `loo` or WAIC `waic` cross-validation
    """
    if len(idatas) < 2:
        raise ValueError("You should provide a list with at least two InferenceData objects")

    if not all("posterior_predictive" in idata.groups() for idata in idatas):
        raise ValueError(
            "All the InferenceData objects must contain the `posterior_predictive` group"
        )

    if not all(idatas[0].observed_data.equals(idata.observed_data) for idata in idatas[1:]):
        raise ValueError("The observed data should be the same for all InferenceData objects")

    if weights is None:
        weights = np.ones(len(idatas)) / len(idatas)
    elif len(idatas) != len(weights):
        raise ValueError(
            "The number of weights should be the same as the number of InferenceData objects"
        )

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    len_idatas = [
        idata.posterior_predictive.dims["chain"] * idata.posterior_predictive.dims["draw"]
        for idata in idatas
    ]

    if not all(len_idatas):
        raise ValueError("At least one of your idatas has 0 samples")

    new_samples = (np.min(len_idatas) * weights).astype(int)

    new_idatas = [
        extract(idata, group="posterior_predictive", num_samples=samples).reset_coords()
        for samples, idata in zip(new_samples, idatas)
    ]

    weighted_samples = InferenceData(
        posterior_predictive=xr.concat(new_idatas, dim="sample"),
        observed_data=idatas[0].observed_data,
    )

    return weighted_samples
