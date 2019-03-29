"""Stats-utility functions for ArviZ."""
from collections.abc import Sequence
import warnings

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats.mstats import mquantiles


__all__ = ["autocorr", "make_ufunc"]


def autocorr(ary):
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
    ary = ary - ary.mean()
    n = len(ary)
    with warnings.catch_warnings():
        # silence annoying numpy tuple warning in another library
        # silence hack added in 0.3.3+
        warnings.simplefilter("ignore")
        result = fftconvolve(ary, ary[::-1])
    acorr = result[len(result) // 2 :]
    acorr /= np.arange(n, 0, -1)
    with np.errstate(invalid="ignore"):
        acorr /= acorr[0]
    return acorr


def _autocov(ary):
    """Compute autocovariance estimates for every lag for the input array.

    Parameters
    ----------
    ary : Numpy array
        An array containing MCMC samples

    Returns
    -------
    acov: Numpy array same size as the input array
    """
    acorr = autocorr(ary)
    varx = np.var(ary, ddof=0)
    acov = acorr * varx
    return acov


def make_ufunc(func, n_dims=2, n_output=1, index=Ellipsis, ravel=True):  # noqa: D202
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
    index : int, optional
        Slice ndarray with `index`. Defaults to `Ellipsis`.
    ravel : bool, optional
        If true, ravel the ndarray before calling `func`.

    Returns
    -------
    callable
        ufunc wrapper for `func`.
    """
    if n_dims < 1:
        raise TypeError("n_dims must be one or higher.")

    def _ufunc(ary, *args, out=None, **kwargs):
        """General ufunc for single-output function."""
        if out is None:
            out = np.empty(ary.shape[:-n_dims])
        else:
            if out.shape != ary.shape[:-n_dims]:
                msg = "Shape incorrect for `out`: {}.".format(out.shape)
                msg += " Correct shape is {}".format(ary.shape[:-n_dims])
                raise TypeError(msg)
        for idx in np.ndindex(out.shape):
            ary_idx = ary[idx].ravel() if ravel else ary[idx]
            out[idx] = np.asarray(func(ary_idx, *args, **kwargs))[index]
        return out

    def _multi_ufunc(ary, *args, out=None, **kwargs):
        """General ufunc for multi-output function."""
        element_shape = ary.shape[:-n_dims]
        if out is None:
            out = tuple(np.empty(element_shape) for _ in range(n_output))
        else:
            raise_error = False
            if isinstance(out, tuple):
                out_shape = tuple(item.shape for item in out)
                correct_shape = tuple(element_shape for _ in range(n_output))
                if out_shape != correct_shape:
                    raise_error = True
            else:
                raise_error = True
            if raise_error:
                msg = "Shapes incorrect for `out`: {}.".format(out_shape)
                msg += " Correct shapes are {}".format(correct_shape)
                raise TypeError(msg)
        for idx in np.ndindex(element_shape):
            ary_idx = ary[idx].ravel() if ravel else ary[idx]
            results = func(ary_idx, *args, **kwargs)
            for i, res in enumerate(results):
                out[i][idx] = np.asarray(res)[index]
        return out

    if n_output > 1:
        ufunc = _multi_ufunc
    else:
        ufunc = _ufunc

    update_docstring(ufunc, func, n_output)
    return ufunc


def update_docstring(ufunc, func, n_output=1):
    """Update ArviZ generated ufunc docstring."""
    module = ""
    name = ""
    docstring = ""
    if hasattr(func, "__module__"):
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


def _rint(num):
    """Round and change to ingeter."""
    rnum = np.rint(num)
    return int(rnum)


def _round(num, decimals):
    """Skip rounding if decimals is None."""
    if decimals is not None:
        num = np.round(num, decimals)
    return num


def _quantile(ary, q, axis=None, limit=None):
    """Use same quantile function as R (Type 7)."""
    if limit is None:
        limit = tuple()
    return mquantiles(ary, q, alphap=1, betap=1, axis=axis, limit=limit)


def check_nan(ary, axis=None, how="any"):
    """Check if ary has NaN values."""
    isnan = np.isnan(ary)
    if how.lower() == "any":
        isnan = isnan.any(axis)
    elif how.lower() == "all":
        isnan = isnan.all(axis)
    return isnan


def check_valid_size(ary, msg):
    """Validate 2D array shape."""
    ary = np.asarray(ary)
    shape = ary.shape
    if len(shape) != 2:
        raise TypeError("{} calculation requires 2 dimensional array.".format(msg))
    n_chain, n_draw = shape
    if n_chain <= 1:
        raise TypeError("{} calculation requires multiple chains.".format(msg))
    if n_draw <= 1:
        raise TypeError("{} calculation requires multiple draws.".format(msg))
