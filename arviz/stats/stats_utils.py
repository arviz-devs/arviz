"""Stats-utility functions for ArviZ."""
import warnings
from collections.abc import Sequence
from copy import copy as _copy
from copy import deepcopy as _deepcopy

import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.stats.mstats import mquantiles
from xarray import apply_ufunc

from .. import _log
from ..utils import conditional_jit, conditional_vect, conditional_dask
from .density_utils import histogram as _histogram


__all__ = ["autocorr", "autocov", "ELPDData", "make_ufunc", "wrap_xarray_ufunc"]


def autocov(ary, axis=-1):
    """Compute autocovariance estimates for every lag for the input array.

    Parameters
    ----------
    ary : Numpy array
        An array containing MCMC samples

    Returns
    -------
    acov: Numpy array same size as the input array
    """
    axis = axis if axis > 0 else len(ary.shape) + axis
    n = ary.shape[axis]
    m = next_fast_len(2 * n)

    ary = ary - ary.mean(axis, keepdims=True)

    # added to silence tuple warning for a submodule
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ifft_ary = np.fft.rfft(ary, n=m, axis=axis)
        ifft_ary *= np.conjugate(ifft_ary)

        shape = tuple(
            slice(None) if dim_len != axis else slice(0, n) for dim_len, _ in enumerate(ary.shape)
        )
        cov = np.fft.irfft(ifft_ary, n=m, axis=axis)[shape]
        cov /= n

    return cov


def autocorr(ary, axis=-1):
    """Compute autocorrelation using FFT for every lag for the input array.

    See https://en.wikipedia.org/wiki/autocorrelation#Efficient_computation

    Parameters
    ----------
    ary : Numpy array
        An array containing MCMC samples

    Returns
    -------
    acorr: Numpy array same size as the input array
    """
    corr = autocov(ary, axis=axis)
    axis = axis = axis if axis > 0 else len(corr.shape) + axis
    norm = tuple(
        slice(None, None) if dim != axis else slice(None, 1) for dim, _ in enumerate(corr.shape)
    )
    with np.errstate(invalid="ignore"):
        corr /= corr[norm]
    return corr


def make_ufunc(
    func, n_dims=2, n_output=1, n_input=1, index=Ellipsis, ravel=True, check_shape=None
):  # noqa: D202
    """Make ufunc from a function taking 1D array input.

    Parameters
    ----------
    func : callable
    n_dims : int, optional
        Number of core dimensions not broadcasted. Dimensions are skipped from the end.
        At minimum n_dims > 0.
    n_output : int, optional
        Select number of results returned by `func`.
        If n_output > 1, ufunc returns a tuple of objects else returns an object.
    n_input : int, optional
        Number of **array** inputs to func, i.e. ``n_input=2`` means that func is called
        with ``func(ary1, ary2, *args, **kwargs)``
    index : int, optional
        Slice ndarray with `index`. Defaults to `Ellipsis`.
    ravel : bool, optional
        If true, ravel the ndarray before calling `func`.
    check_shape: bool, optional
        If false, do not check if the shape of the output is compatible with n_dims and
        n_output. By default, True only for n_input=1. If n_input is larger than 1, the last
        input array is used to check the shape, however, shape checking with multiple inputs
        may not be correct.

    Returns
    -------
    callable
        ufunc wrapper for `func`.
    """
    if n_dims < 1:
        raise TypeError("n_dims must be one or higher.")

    if n_input == 1 and check_shape is None:
        check_shape = True
    elif check_shape is None:
        check_shape = False

    def _ufunc(*args, out=None, out_shape=None, **kwargs):
        """General ufunc for single-output function."""
        arys = args[:n_input]
        n_dims_out = None
        if out is None:
            if out_shape is None:
                out = np.empty(arys[-1].shape[:-n_dims])
            else:
                out = np.empty((*arys[-1].shape[:-n_dims], *out_shape))
                n_dims_out = -len(out_shape)
        elif check_shape:
            if out.shape != arys[-1].shape[:-n_dims]:
                msg = f"Shape incorrect for `out`: {out.shape}."
                msg += f" Correct shape is {arys[-1].shape[:-n_dims]}"
                raise TypeError(msg)
        for idx in np.ndindex(out.shape[:n_dims_out]):
            arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
            out[idx] = np.asarray(func(*arys_idx, *args[n_input:], **kwargs))[index]
        return out

    def _multi_ufunc(*args, out=None, out_shape=None, **kwargs):
        """General ufunc for multi-output function."""
        arys = args[:n_input]
        element_shape = arys[-1].shape[:-n_dims]
        if out is None:
            if out_shape is None:
                out = tuple(np.empty(element_shape) for _ in range(n_output))
            else:
                out = tuple(np.empty((*element_shape, *out_shape[i])) for i in range(n_output))

        elif check_shape:
            raise_error = False
            correct_shape = tuple(element_shape for _ in range(n_output))
            if isinstance(out, tuple):
                out_shape = tuple(item.shape for item in out)
                if out_shape != correct_shape:
                    raise_error = True
            else:
                raise_error = True
                out_shape = "not tuple, type={type(out)}"
            if raise_error:
                msg = f"Shapes incorrect for `out`: {out_shape}."
                msg += f" Correct shapes are {correct_shape}"
                raise TypeError(msg)
        for idx in np.ndindex(element_shape):
            arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
            results = func(*arys_idx, *args[n_input:], **kwargs)
            for i, res in enumerate(results):
                out[i][idx] = np.asarray(res)[index]
        return out

    if n_output > 1:
        ufunc = _multi_ufunc
    else:
        ufunc = _ufunc

    update_docstring(ufunc, func, n_output)
    return ufunc


@conditional_dask
def wrap_xarray_ufunc(
    ufunc,
    *datasets,
    ufunc_kwargs=None,
    func_args=None,
    func_kwargs=None,
    dask_kwargs=None,
    **kwargs,
):
    """Wrap make_ufunc with xarray.apply_ufunc.

    Parameters
    ----------
    ufunc : callable
    *datasets : xarray.Dataset
    ufunc_kwargs : dict
        Keyword arguments passed to `make_ufunc`.
            - 'n_dims', int, by default 2
            - 'n_output', int, by default 1
            - 'n_input', int, by default len(datasets)
            - 'index', slice, by default Ellipsis
            - 'ravel', bool, by default True
    func_args : tuple
        Arguments passed to 'ufunc'.
    func_kwargs : dict
        Keyword arguments passed to 'ufunc'.
            - 'out_shape', int, by default None
    dask_kwargs : dict
        Dask related kwargs passed to :func:`xarray:xarray.apply_ufunc`.
        Use ``enable_dask`` method of :class:`arviz.Dask` to set default kwargs.
    **kwargs
        Passed to :func:`xarray.apply_ufunc`.

    Returns
    -------
    xarray.Dataset
    """
    if ufunc_kwargs is None:
        ufunc_kwargs = {}
    ufunc_kwargs.setdefault("n_input", len(datasets))
    if func_args is None:
        func_args = tuple()
    if func_kwargs is None:
        func_kwargs = {}
    if dask_kwargs is None:
        dask_kwargs = {}

    kwargs.setdefault(
        "input_core_dims", tuple(("chain", "draw") for _ in range(len(func_args) + len(datasets)))
    )
    ufunc_kwargs.setdefault("n_dims", len(kwargs["input_core_dims"][-1]))
    kwargs.setdefault("output_core_dims", tuple([] for _ in range(ufunc_kwargs.get("n_output", 1))))

    callable_ufunc = make_ufunc(ufunc, **ufunc_kwargs)

    return apply_ufunc(
        callable_ufunc, *datasets, *func_args, kwargs=func_kwargs, **dask_kwargs, **kwargs
    )


def update_docstring(ufunc, func, n_output=1):
    """Update ArviZ generated ufunc docstring."""
    module = ""
    name = ""
    docstring = ""
    if hasattr(func, "__module__") and isinstance(func.__module__, str):
        module += func.__module__
    if hasattr(func, "__name__"):
        name += func.__name__
    if hasattr(func, "__doc__") and isinstance(func.__doc__, str):
        docstring += func.__doc__
    ufunc.__doc__ += "\n\n"
    if module or name:
        ufunc.__doc__ += "This function is a ufunc wrapper for "
        ufunc.__doc__ += module + "." + name
        ufunc.__doc__ += "\n"
    ufunc.__doc__ += 'Call ufunc with n_args from xarray against "chain" and "draw" dimensions:'
    ufunc.__doc__ += "\n\n"
    input_core_dims = 'tuple(("chain", "draw") for _ in range(n_args))'
    if n_output > 1:
        output_core_dims = f" tuple([] for _ in range({n_output}))"
        msg = f"xr.apply_ufunc(ufunc, dataset, input_core_dims={input_core_dims}, "
        msg += f"output_core_dims={ output_core_dims})"
    else:
        output_core_dims = ""
        msg = f"xr.apply_ufunc(ufunc, dataset, input_core_dims={input_core_dims})"
    ufunc.__doc__ += msg
    ufunc.__doc__ += "\n\n"
    ufunc.__doc__ += "For example: np.std(data, ddof=1) --> n_args=2"
    if docstring:
        ufunc.__doc__ += "\n\n"
        ufunc.__doc__ += module
        ufunc.__doc__ += name
        ufunc.__doc__ += " docstring:"
        ufunc.__doc__ += "\n\n"
        ufunc.__doc__ += docstring


def logsumexp(ary, *, b=None, b_inv=None, axis=None, keepdims=False, out=None, copy=True):
    """Stable logsumexp when b >= 0 and b is scalar.

    b_inv overwrites b unless b_inv is None.
    """
    # check dimensions for result arrays
    ary = np.asarray(ary)
    if ary.dtype.kind == "i":
        ary = ary.astype(np.float64)
    dtype = ary.dtype.type
    shape = ary.shape
    shape_len = len(shape)
    if isinstance(axis, Sequence):
        axis = tuple(axis_i if axis_i >= 0 else shape_len + axis_i for axis_i in axis)
        agroup = axis
    else:
        axis = axis if (axis is None) or (axis >= 0) else shape_len + axis
        agroup = (axis,)
    shape_max = (
        tuple(1 for _ in shape)
        if axis is None
        else tuple(1 if i in agroup else d for i, d in enumerate(shape))
    )
    # create result arrays
    if out is None:
        if not keepdims:
            out_shape = (
                tuple()
                if axis is None
                else tuple(d for i, d in enumerate(shape) if i not in agroup)
            )
        else:
            out_shape = shape_max
        out = np.empty(out_shape, dtype=dtype)
    if b_inv == 0:
        return np.full_like(out, np.inf, dtype=dtype) if out.shape else np.inf
    if b_inv is None and b == 0:
        return np.full_like(out, -np.inf) if out.shape else -np.inf
    ary_max = np.empty(shape_max, dtype=dtype)
    # calculations
    ary.max(axis=axis, keepdims=True, out=ary_max)
    if copy:
        ary = ary.copy()
    ary -= ary_max
    np.exp(ary, out=ary)
    ary.sum(axis=axis, keepdims=keepdims, out=out)
    np.log(out, out=out)
    if b_inv is not None:
        ary_max -= np.log(b_inv)
    elif b:
        ary_max += np.log(b)
    out += ary_max if keepdims else ary_max.squeeze()
    # transform to scalar if possible
    return out if out.shape else dtype(out)


def quantile(ary, q, axis=None, limit=None):
    """Use same quantile function as R (Type 7)."""
    if limit is None:
        limit = tuple()
    return mquantiles(ary, q, alphap=1, betap=1, axis=axis, limit=limit)


def not_valid(ary, check_nan=True, check_shape=True, nan_kwargs=None, shape_kwargs=None):
    """Validate ndarray.

    Parameters
    ----------
    ary : numpy.ndarray
    check_nan : bool
        Check if any value contains NaN.
    check_shape : bool
        Check if array has correct shape. Assumes dimensions in order (chain, draw, *shape).
        For 1D arrays (shape = (n,)) assumes chain equals 1.
    nan_kwargs : dict
        Valid kwargs are:
            axis : int,
                Defaults to None.
            how : str, {"all", "any"}
                Default to "any".
    shape_kwargs : dict
        Valid kwargs are:
            min_chains : int
                Defaults to 1.
            min_draws : int
                Defaults to 4.

    Returns
    -------
    bool
    """
    ary = np.asarray(ary)

    nan_error = False
    draw_error = False
    chain_error = False

    if check_nan:
        if nan_kwargs is None:
            nan_kwargs = {}

        isnan = np.isnan(ary)
        axis = nan_kwargs.get("axis", None)
        if nan_kwargs.get("how", "any").lower() == "all":
            nan_error = isnan.all(axis)
        else:
            nan_error = isnan.any(axis)

        if (isinstance(nan_error, bool) and nan_error) or nan_error.any():
            _log.warning("Array contains NaN-value.")

    if check_shape:
        shape = ary.shape

        if shape_kwargs is None:
            shape_kwargs = {}

        min_chains = shape_kwargs.get("min_chains", 2)
        min_draws = shape_kwargs.get("min_draws", 4)
        error_msg = f"Shape validation failed: input_shape: {shape}, "
        error_msg += f"minimum_shape: (chains={min_chains}, draws={min_draws})"

        chain_error = ((min_chains > 1) and (len(shape) < 2)) or (shape[0] < min_chains)
        draw_error = ((len(shape) < 2) and (shape[0] < min_draws)) or (
            (len(shape) > 1) and (shape[1] < min_draws)
        )

        if chain_error or draw_error:
            _log.warning(error_msg)

    return nan_error | chain_error | draw_error


def get_log_likelihood(idata, var_name=None):
    """Retrieve the log likelihood dataarray of a given variable."""
    if (
        not hasattr(idata, "log_likelihood")
        and hasattr(idata, "sample_stats")
        and hasattr(idata.sample_stats, "log_likelihood")
    ):
        warnings.warn(
            "Storing the log_likelihood in sample_stats groups has been deprecated",
            DeprecationWarning,
        )
        return idata.sample_stats.log_likelihood
    if not hasattr(idata, "log_likelihood"):
        raise TypeError("log likelihood not found in inference data object")
    if var_name is None:
        var_names = list(idata.log_likelihood.data_vars)
        if len(var_names) > 1:
            raise TypeError(
                f"Found several log likelihood arrays {var_names}, var_name cannot be None"
            )
        return idata.log_likelihood[var_names[0]]
    else:
        try:
            log_likelihood = idata.log_likelihood[var_name]
        except KeyError as err:
            raise TypeError(f"No log likelihood data named {var_name} found") from err
        return log_likelihood


BASE_FMT = """Computed from {{n_samples}} posterior samples and \
{{n_points}} observations log-likelihood matrix.

{{0:{0}}} Estimate       SE
{{scale}}_{{kind}} {{1:8.2f}}  {{2:7.2f}}
p_{{kind:{1}}} {{3:8.2f}}        -"""
POINTWISE_LOO_FMT = """------

Pareto k diagnostic values:
                         {{0:>{0}}} {{1:>6}}
(-Inf, 0.5]   (good)     {{2:{0}d}} {{6:6.1f}}%
 (0.5, 0.7]   (ok)       {{3:{0}d}} {{7:6.1f}}%
   (0.7, 1]   (bad)      {{4:{0}d}} {{8:6.1f}}%
   (1, Inf)   (very bad) {{5:{0}d}} {{9:6.1f}}%
"""
SCALE_DICT = {"deviance": "deviance", "log": "elpd", "negative_log": "-elpd"}


class ELPDData(pd.Series):  # pylint: disable=too-many-ancestors
    """Class to contain the data from elpd information criterion like waic or loo."""

    def __str__(self):
        """Print elpd data in a user friendly way."""
        kind = self.index[0].split("_")[1]

        if kind not in ("loo", "waic"):
            raise ValueError("Invalid ELPDData object")

        scale_str = SCALE_DICT[self["scale"]]
        padding = len(scale_str) + len(kind) + 1
        base = BASE_FMT.format(padding, padding - 2)
        base = base.format(
            "",
            kind=kind,
            scale=scale_str,
            n_samples=self.n_samples,
            n_points=self.n_data_points,
            *self.values,
        )

        if self.warning:
            base += "\n\nThere has been a warning during the calculation. Please check the results."

        if kind == "loo" and "pareto_k" in self:
            bins = np.asarray([-np.Inf, 0.5, 0.7, 1, np.Inf])
            counts, *_ = _histogram(self.pareto_k.values, bins)
            extended = POINTWISE_LOO_FMT.format(max(4, len(str(np.max(counts)))))
            extended = extended.format(
                "Count", "Pct.", *[*counts, *(counts / np.sum(counts) * 100)]
            )
            base = "\n".join([base, extended])
        return base

    def __repr__(self):
        """Alias to ``__str__``."""
        return self.__str__()

    def copy(self, deep=True):  # pylint:disable=overridden-final-method
        """Perform a pandas deep copy of the ELPDData plus a copy of the stored data."""
        copied_obj = pd.Series.copy(self)
        for key in copied_obj.keys():
            if deep:
                copied_obj[key] = _deepcopy(copied_obj[key])
            else:
                copied_obj[key] = _copy(copied_obj[key])
        return ELPDData(copied_obj)


@conditional_jit
def stats_variance_1d(data, ddof=0):
    a_a, b_b = 0, 0
    for i in data:
        a_a = a_a + i
        b_b = b_b + i * i
    var = b_b / (len(data)) - ((a_a / (len(data))) ** 2)
    var = var * (len(data) / (len(data) - ddof))
    return var


def stats_variance_2d(data, ddof=0, axis=1):
    if data.ndim == 1:
        return stats_variance_1d(data, ddof=ddof)
    a_a, b_b = data.shape
    if axis == 1:
        var = np.zeros(a_a)
        for i in range(a_a):
            var[i] = stats_variance_1d(data[i], ddof=ddof)
    else:
        var = np.zeros(b_b)
        for i in range(b_b):
            var[i] = stats_variance_1d(data[:, i], ddof=ddof)

    return var


@conditional_vect
def _sqrt(a_a, b_b):
    return (a_a + b_b) ** 0.5


def _circfunc(samples, high, low, skipna):
    samples = np.asarray(samples)
    if skipna:
        samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return np.nan
    return _angle(samples, low, high, np.pi)


@conditional_vect
def _angle(samples, low, high, p_i=np.pi):
    ang = (samples - low) * 2.0 * p_i / (high - low)
    return ang


def _circular_standard_deviation(samples, high=2 * np.pi, low=0, skipna=False, axis=None):
    ang = _circfunc(samples, high, low, skipna)
    s_s = np.sin(ang).mean(axis=axis)
    c_c = np.cos(ang).mean(axis=axis)
    r_r = np.hypot(s_s, c_c)
    return ((high - low) / 2.0 / np.pi) * np.sqrt(-2 * np.log(r_r))


def smooth_data(obs_vals, pp_vals):
    """Smooth data, helper function for discrete data in plot_pbv, loo_pit and plot_loo_pit."""
    x = np.linspace(0, 1, len(obs_vals))
    csi = CubicSpline(x, obs_vals)
    obs_vals = csi(np.linspace(0.01, 0.99, len(obs_vals)))

    x = np.linspace(0, 1, pp_vals.shape[1])
    csi = CubicSpline(x, pp_vals, axis=1)
    pp_vals = csi(np.linspace(0.01, 0.99, pp_vals.shape[1]))

    return obs_vals, pp_vals
