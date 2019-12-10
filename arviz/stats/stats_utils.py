"""Stats-utility functions for ArviZ."""
from collections.abc import Sequence
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.stats.mstats import mquantiles
from xarray import apply_ufunc
from ..utils import conditional_jit

_log = logging.getLogger(__name__)

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

    def _ufunc(*args, out=None, **kwargs):
        """General ufunc for single-output function."""
        arys = args[:n_input]
        if out is None:
            out = np.empty(arys[-1].shape[:-n_dims])
        elif check_shape:
            if out.shape != arys[-1].shape[:-n_dims]:
                msg = "Shape incorrect for `out`: {}.".format(out.shape)
                msg += " Correct shape is {}".format(arys[-1].shape[:-n_dims])
                raise TypeError(msg)
        for idx in np.ndindex(out.shape):
            arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
            out[idx] = np.asarray(func(*arys_idx, *args[n_input:], **kwargs))[index]
        return out

    def _multi_ufunc(*args, out=None, **kwargs):
        """General ufunc for multi-output function."""
        arys = args[:n_input]
        element_shape = arys[-1].shape[:-n_dims]
        if out is None:
            out = tuple(np.empty(element_shape) for _ in range(n_output))
        elif check_shape:
            raise_error = False
            correct_shape = tuple(element_shape for _ in range(n_output))
            if isinstance(out, tuple):
                out_shape = tuple(item.shape for item in out)
                if out_shape != correct_shape:
                    raise_error = True
            else:
                raise_error = True
                out_shape = "not tuple, type={}".format(type(out))
            if raise_error:
                msg = "Shapes incorrect for `out`: {}.".format(out_shape)
                msg += " Correct shapes are {}".format(correct_shape)
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


def wrap_xarray_ufunc(
    ufunc, *datasets, ufunc_kwargs=None, func_args=None, func_kwargs=None, **kwargs
):
    """Wrap make_ufunc with xarray.apply_ufunc.

    Parameters
    ----------
    ufunc : callable
    datasets : xarray.dataset
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
    **kwargs
        Passed to xarray.apply_ufunc.

    Returns
    -------
    xarray.dataset
    """
    if ufunc_kwargs is None:
        ufunc_kwargs = {}
    ufunc_kwargs.setdefault("n_input", len(datasets))
    if func_args is None:
        func_args = tuple()
    if func_kwargs is None:
        func_kwargs = {}

    callable_ufunc = make_ufunc(ufunc, **ufunc_kwargs)

    kwargs.setdefault(
        "input_core_dims", tuple(("chain", "draw") for _ in range(len(func_args) + 1))
    )
    kwargs.setdefault("output_core_dims", tuple([] for _ in range(ufunc_kwargs.get("n_output", 1))))

    return apply_ufunc(callable_ufunc, *datasets, *func_args, kwargs=func_kwargs, **kwargs)


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
        output_core_dims = " tuple([] for _ in range({}))".format(n_output)
        msg = "xr.apply_ufunc(ufunc, dataset, input_core_dims={}, output_core_dims={})"
        ufunc.__doc__ += msg.format(input_core_dims, output_core_dims)
    else:
        output_core_dims = ""
        msg = "xr.apply_ufunc(ufunc, dataset, input_core_dims={})"
        ufunc.__doc__ += msg.format(input_core_dims)
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
    out += ary_max.squeeze() if not keepdims else ary_max
    # transform to scalar if possible
    return out if out.shape else dtype(out)


def rint(num):
    """Round and change to ingeter."""
    rnum = np.rint(num)  # pylint: disable=assignment-from-no-return
    return int(rnum)


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
            nan_kwargs = dict()

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
            shape_kwargs = dict()

        min_chains = shape_kwargs.get("min_chains", 2)
        min_draws = shape_kwargs.get("min_draws", 4)
        error_msg = "Shape validation failed: input_shape: {}, minimum_shape: (chains={}, draws={})"
        error_msg = error_msg.format(shape, min_chains, min_draws)

        chain_error = ((min_chains > 1) and (len(shape) < 2)) or (shape[0] < min_chains)
        draw_error = ((len(shape) < 2) and (shape[0] < min_draws)) or (
            (len(shape) > 1) and (shape[1] < min_draws)
        )

        if chain_error or draw_error:
            _log.warning(error_msg)

    return nan_error | chain_error | draw_error


BASE_FMT = """Computed from {{n_samples}} by {{n_points}} log-likelihood matrix

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
SCALE_DICT = {"deviance": "IC", "log": "elpd", "negative_log": "-elpd"}


class ELPDData(pd.Series):  # pylint: disable=too-many-ancestors
    """Class to contain the data from elpd information criterion like waic or loo."""

    def __str__(self):
        """Print elpd data in a user friendly way."""
        kind = self.index[0]

        if kind not in ("waic", "loo"):
            raise ValueError("Invalid ELPDData object")

        scale_str = SCALE_DICT[self["{}_scale".format(kind)]]
        padding = len(scale_str) + len(kind) + 1
        base = BASE_FMT.format(padding, padding - 2)
        base = base.format(
            "",
            kind=kind,
            scale=scale_str,
            n_samples=self.n_samples,
            n_points=self.n_data_points,
            *self.values
        )

        if self.warning:
            base += "\n\nThere has been a warning during the calculation. Please check the results."

        if kind == "loo" and "pareto_k" in self:
            bins = np.asarray([-np.Inf, 0.5, 0.7, 1, np.Inf])
            counts, *_ = histogram(self.pareto_k.values, bins)
            extended = POINTWISE_LOO_FMT.format(max(4, len(str(np.max(counts)))))
            extended = extended.format(
                "Count", "Pct.", *[*counts, *(counts / np.sum(counts) * 100)]
            )
            base = "\n".join([base, extended])
        return base

    def __repr__(self):
        """Alias to ``__str__``."""
        return self.__str__()


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
        return var
    else:
        var = np.zeros(b_b)
        for i in range(b_b):
            var[i] = stats_variance_1d(data[:, i], ddof=ddof)
        return var


@conditional_jit(cache=True)
def histogram(data, bins, range_hist=None):
    """Conditionally jitted histogram.

    Parameters
    ----------
    data : array-like
        Input data. Passed as first positional argument to ``np.histogram``.
    bins : int or array-like
        Passed as keyword argument ``bins`` to ``np.histogram``.
    range_hist : (float, float), optional
        Passed as keyword argument ``range`` to ``np.histogram``.

    Returns
    -------
    hist : array
        The number of counts per bin.
    density : array
        The density corresponding to each bin.
    bin_edges : array
        The edges of the bins used.
    """
    hist, bin_edges = np.histogram(data, bins=bins, range=range_hist)
    hist_dens = hist / (hist.sum() * np.diff(bin_edges))
    return hist, hist_dens, bin_edges
