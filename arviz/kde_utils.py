"""Kernel density estimation functions for ArviZ."""

import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.signal import gaussian, convolve
from scipy.special import ive

from .stats.stats_utils import histogram

# Bandwidth functions ---------------------------------------------------------------
# Linear KDE
def bw_scott(x, x_std=None):
    """
    Scott's Rule
    """
    if x_std is None:
        x_std = np.std(x)
    bw = 1.06 * x_std * len(x) ** (-0.2)
    return bw


def bw_silverman(x, x_std=None):
    """
    Silverman's Rule.
    """
    if x_std is None:
        x_std = np.std(x)
    q75, q25 = np.percentile(x, [75, 25])
    x_iqr = q75 - q25
    a = min(x_std, x_iqr / 1.34)
    bw = 0.9 * a * len(x) ** (-0.2)
    return bw


def bw_isj(x, grid_counts=None, x_range=None):
    """
    Improved Sheather and Jones method as explained in [1]
    This is an internal version pretended to be used by the KDE estimator.
    When used internally computation time is saved because things like minimums,

    maximums and the grid are pre-computed.

    References
    ----------
    .. [1] Kernel density estimation via diffusion.
       Z. I. Botev, J. F. Grotowski, and D. P. Kroese.
       Ann. Statist. 38 (2010), no. 5, 2916--2957.
    """

    x_len = len(x)
    if x_range is None:
        x_min = np.min(x)
        x_max = np.max(x)
        x_range = x_max - x_min

    # Relative frequency per bin
    if grid_counts is None:
        x_std = np.std(x)
        grid_len = 256
        grid_min = x_min - 0.5 * x_std
        grid_max = x_max + 0.5 * x_std
        grid_counts, _, _ = histogram(x, grid_len, (grid_min, grid_max))
    else:
        grid_len = len(grid_counts) - 1

    grid_relfreq = grid_counts / x_len

    # Discrete cosine transform of the data
    a_k = dct1d(grid_relfreq)

    k_sq = np.arange(1, grid_len) ** 2
    a_sq = a_k[range(1, grid_len)] ** 2

    t = _root(_fixed_point, x_len, args=(x_len, k_sq, a_sq), x=x)
    h = t ** 0.5 * x_range
    return h

def bw_experimental(x, grid_counts=None, x_std=None, x_range=None):
    """
    Experimental bandwidth estimator.
    """
    return 0.5 * (bw_silverman(x, x_std) + bw_isj(x, grid_counts, x_range))


def bw_taylor(x):
    """
    Bandwidth selector for circular kernel density estimation
    as introduced in [1].
    This function implements a rule-of-thumb for choosing the bandwidth of
    a von Mises kernel density estimator that assumes the underlying
    distribution is von Mises.
    It is analogous to Scott's rule for the Gaussian KDE.

    Circular bandwidth has a different scale from linear bandwidth.
    Unlike linear scale, low bandwidths are associated with oversmoothing
    while high values are associated with undersmoothing.

    References
    ----------
    [1] C.C Taylor (2008). Automatic bandwidth selection
    for circular density estimation.
    Computational Statistics and Data Analysis, 52, 7, 3493–3500.
    """
    x_len = len(x)
    kappa = kappa_mle(x)
    num = 3 * x_len * kappa ** 2 * ive(2, 2 * kappa)
    den = 4 * np.pi ** 0.5 * ive(0, kappa) ** 2
    return (num / den) ** 0.4


BW_METHODS = {
        "scott": bw_scott,
        "silverman": bw_silverman,
        "isj": bw_isj,
        "experimental": bw_experimental
    }


def select_bw_method(method):
    """
    Selects a function to compute the bandwidth.
    Also checks method `bw` is correctly specified.
    Otherwise, throws an error.

    Parameters
    ----------
    method : str
        Method to estimate the bandwidth.

    Returns
    -------
    bw_fun: function
        Function to compute the bandwidth.
    """

    method_lower = method.lower()
    if method_lower not in BW_METHODS.keys():
        raise ValueError((
            f"Unrecognized bandwidth method.\n"
            f"Input is: {method}.\n"
            f"Expected one of: {list(BW_METHODS.keys())}."
        ))
    bw_fun = BW_METHODS[method_lower]
    return bw_fun


def get_bw(x, bw, grid_counts=None, x_std=None, x_range=None):
    """
    Computes bandwidth for a given data `x` and `bw`.
    Also checks `bw` is correctly specified.

    Parameters
    ----------
    x : 1-D numpy array
        1 dimensional array of sample data from the
        variable for which a density estimate is desired.
    bw: int, float or str
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth.

    Returns
    -------
    bw: float
        Bandwidth
    """
    if isinstance(bw, bool):
        raise ValueError((
            f"`bw` must not be of type `bool`.\n"
            f"Expected a positive numeric or one of the following strings:\n"
            f"{list(BW_METHODS.keys())}."))
    if isinstance(bw, (int, float)):
        if bw < 0:
            raise ValueError(f"Numeric `bw` must be positive.\nInput: {bw:.4f}.")
    elif isinstance(bw, str):
        bw_fun = select_bw_method(bw)
        bw_lower = bw.lower()
        if bw_lower == "isj":
            bw = bw_fun(x, grid_counts, x_range)
        elif bw_lower in ["scott", "silverman"]:
            bw = bw_fun(x, x_std)
        elif bw_lower == "experimental":
            bw = bw_fun(x, grid_counts, x_std, x_range)
    else:
        raise ValueError((
            f"Unrecognized `bw` argument.\n"
            f"Expected a positive numeric or one of the following strings:\n"
            f"{list(BW_METHODS.keys())}."))
    return bw


# Misc utils
def vonmises_pdf(x, mu, kappa):
    assert kappa > 0, "Argument 'kappa' must be positive."
    pdf = 1 / (2 * np.pi * ive(0, kappa)) * np.exp(np.cos(x - mu) - 1) ** kappa
    return pdf


def circular_mean(x):
    sinr = np.sum(np.sin(x))
    cosr = np.sum(np.cos(x))
    mean = np.arctan2(sinr, cosr)
    return mean


def a1inv(x):
    """
    Inverse function of the ratio of the first and zeroth order
    Bessel functions of the first kind.

    Returns the value k, such that a1inv(x) = k, i.e. a1(k) = x.
    """
    if 0 <= x < 0.53:
        return 2 * x + x ** 3 + (5 * x ** 5) / 6
    elif x < 0.85:
        return -0.4 + 1.39 * x + 0.43 / (1 - x)
    else:
        return 1 / (x ** 3 - 4 * x ** 2 + 3 * x)


def kappa_mle(x):
    mean = circular_mean(x)
    kappa = a1inv(np.mean(np.cos(x - mean)))
    return kappa


def dct1d(x):
    """
    Discrete Cosine Transform in 1 Dimension

    Parameters
    ----------
    x : numpy array
        1 dimensional array of values for which the
        DCT is desired

    Returns
    -------
    output : DTC transformed values
    """

    x_len = len(x)

    even_increasing = np.arange(0, x_len, 2)
    odd_decreasing = np.arange(x_len - 1, 0, -2)

    x = np.concatenate((x[even_increasing], x[odd_decreasing]))

    w_1k = np.r_[
        1,
        (2 * np.exp(-(0 + 1j) * (np.arange(1, x_len)) * np.pi / (2 * x_len)))
    ]
    output = np.real(w_1k * fft(x))

    return output


def _fixed_point(t, N, k_sq, a_sq):
    """
    Implementation of the function t-zeta*gamma^[l](t) derived
    from equation (30) in [1]

    References
    ----------
    .. [1] Kernel density estimation via diffusion.
       Z. I. Botev, J. F. Grotowski, and D. P. Kroese.
       Ann. Statist. 38 (2010), no. 5, 2916--2957.
    """

    k_sq = np.asfarray(k_sq, dtype=np.float64)
    a_sq = np.asfarray(a_sq, dtype=np.float64)

    l = 7
    f = np.sum(np.power(k_sq, l) * a_sq * np.exp(-k_sq * np.pi ** 2 * t))
    f *= 0.5 * np.pi ** (2.0 * l)

    for j in reversed(range(2, l)):
        c1 = (1 + 0.5 ** (j + 0.5)) / 3
        c2 = np.product(np.arange(1.0, 2 * j + 1, 2, dtype=np.float64))
        c2 /= (np.pi / 2) ** 0.5
        t_j = np.power((c1 * (c2 / (N * f))), (2.0 / (3.0 + 2.0 * j)))
        f = np.sum(k_sq ** j * a_sq * np.exp(-k_sq * np.pi ** 2.0 * t_j))
        f *= 0.5 * np.pi ** (2 * j)

    out = t - (2 * N * np.pi ** 0.5 * f) ** (-0.4)
    return out


def _root(function, N, args, x):
    # The idea here was borrowed from KDEpy implementation.
    # The right bound is at most 0.01
    found = 0
    N = max(min(1050, N), 50)
    tol = 10e-12 + 0.01 * (N - 50) / 1000

    while found == 0:
        try:
            bw, res = brentq(function, 0, 0.01, args=args, full_output=True,
                             disp=False)
            found = 1 if res.converged else 0
        except ValueError:
            bw = 0
            tol *= 2.0
            found = 0
        if bw <= 0:
            warnings.warn(
                "Improved Sheather-Jones did not converge to a positive value. "
                "Using Silverman's rule instead.",
                Warning
            )
            bw = (bw_silverman(x) / np.ptp(x)) ** 2
            return bw
        if tol >= 1:
            warnings.warn(
                "Improved Sheather-Jones did not converge. "
                "Using Silverman's rule instead.",
                Warning
            )
            bw = (bw_silverman(x) / np.ptp(x)) ** 2
    return bw

# KDE Utilities ---------------------------------------------------------------------

def check_type(x):
    """
    Checks the input is of the correct type.
    It only accepts numeric lists/numpy arrays of 1 dimension.
    If input is not of the correct type, an informative message is thrown.

    Parameters
    ----------
    x : Object whose type is checked before computing the KDE.

    Returns
    -------
    x : 1-D numpy array
        If no error is thrown, a 1 dimensional array of
        sample data from the variable for which a density estimate is desired.

    """
    if not isinstance(x, (list, np.ndarray)):
        raise ValueError((
            f"`x` is of the wrong type.\n"
            f"Can't produce a density estimator for {type(x)}.\n"
            f"Please input a numeric list or numpy array."
        ))

    # Will raise an error if `x` can't be casted to numeric
    x = np.asfarray(x)

    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("`x` does not contain any finite number.")

    if x.ndim != 1:
        raise ValueError((
            f"Unsupported dimension number.\n"
            f"Density estimator only works with 1-dimensional data, not {x.ndim}."
        ))
    return x


def check_custom_lims(custom_lims, x_min, x_max):
    """
    Checks whether `custom_lims` are of the correct type.
    It accepts numeric lists/tuples of length 2.

    Parameters
    ----------
    custom_lims : Object whose type is checked.

    Returns
    -------
    None: Object of type None

    """
    if not isinstance(custom_lims, (list, tuple)):
        raise TypeError((
            f"`custom_lims` must be a numeric list or tuple of length 2.\n"
            f"Not an object of {type(custom_lims)}."
        ))

    if len(custom_lims) != 2:
        raise AttributeError(
            f"`len(custom_lims)` must be 2, not {len(custom_lims)}.")

    any_bool = any(isinstance(i, bool) for i in custom_lims)
    if any_bool:
        raise TypeError(
            "Elements of `custom_lims` must be numeric or None, not bool.")

    if custom_lims[0] is None:
        custom_lims[0] = x_min

    if custom_lims[1] is None:
        custom_lims[1] = x_max

    all_numeric = all(isinstance(i, (int, float, np.integer, np.float)) for i in custom_lims)
    if not all_numeric:
        raise TypeError((
            f"Elements of `custom_lims` must be numeric or None.\n"
            f"At least one of them is not."
        ))

    if not custom_lims[0] < custom_lims[1]:
        raise AttributeError(
            f"`custom_lims[0]` must be smaller than `custom_lims[1]`.")

    return custom_lims

def get_grid(x_min, x_max, x_std, extend_fct, grid_len, custom_lims, extend=True,
            bound_correction=False):
    """
    Computes the grid that bins the data used to estimate the density function

    Parameters
    ----------
    x_min : float
        Minimum value of the data
    x_max: float
        Maximum value of the data.
    x_std: float
        Standard deviation of the data.
    extend_fct: bool
        Indicates the factor by which `x_std` is multiplied
        to extend the range of the data.
    grid_len: int
        Number of bins
    custom_lims: tuple or list
        Custom limits for the domain of the density estimation.
        Must be numeric of length 2.
    extend: bool, optional
        Whether to extend the range of the data or not.
        Default is True.
    bound_correction: bool, optional
        Whether the density estimations performs boundary correction or not.
        This does not impacts directly in the output, but is used
        to override `extend`.
        Default is False.

    Returns
    -------
    grid_len: int
        Number of bins
    grid_min: float
        Minimum value of the grid
    grid_max: float
        Maximum value of the grid

    """

    # Set up number of bins.
    if grid_len < 100:
        grid_len = 100
    grid_len = int(grid_len)

    # Set up domain
    # `custom_lims` overrides `extend`
    # `bound_correction` overrides `extend`
    if custom_lims is not None:
        custom_lims = check_custom_lims(custom_lims, x_min, x_max)
        grid_min = custom_lims[0]
        grid_max = custom_lims[1]
    elif extend and not bound_correction:
        grid_extend = extend_fct * x_std
        grid_min = x_min - grid_extend
        grid_max = x_max + grid_extend
    else:
        grid_min = x_min
        grid_max = x_max
    return grid_min, grid_max, grid_len

# KDE Functions --------------------------------------------------------------------

def kde(x, circular=False, **kwargs):
    """
    1 dimensional density estimation.
    It is a wrapper around `kde_linear()` and `kde_circular()`.

    Parameters
    ----------
    x : 1D numpy array
        Data used to calculate the density estimation.
        Theoritically it is a random sample obtained from $f$,
        the true probability density function we aim to estimate.
    circular: bool, optional
        Whether `x` is a circular variable or not. Defaults to False.
    **kwargs: Arguments passed to `kde_linear()` and `kde_circular()`.
        See their documentation for more info.

    Returns
    -------
    grid : Gridded numpy array for the x values.
    pdf : Numpy array for the density estimates.
    bw: optional, the estimated bandwidth.
    """
    if circular:
        kde_fun = kde_circular
    else:
        kde_fun = kde_linear

    return kde_fun(x, **kwargs)


def kde_linear(
    x,
    bw="experimental",
    adaptive=False,
    extend=False,
    bound_correction=True,
    extend_fct=0,
    bw_fct=1,
    bw_return=False,
    custom_lims=None,
    cumulative=False,
    grid_len=512
):
    """
    1 dimensional density estimation for linear data.

    Given an array of data points `x` it returns an estimate of
    the probability density function that generated the samples in `x`.

    Parameters
    ----------
    x : 1D numpy array
        Data used to calculate the density estimation.
        Theoritically it is a random sample obtained from $f$,
        the true probability density function we aim to estimate.
    bw: int, float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        one of "scott", "silverman", "isj" or "experimental".
        Defaults to "experimental".
    adaptive: boolean, optional
        Indicates if the bandwidth is adaptative or not.
        It is the recommended approach when there are multiple modalities
        with different spread.
        It is not compatible with convolution. Defaults to False.
    extend: boolean, optional
        Whether to extend the observed range for `x` in the estimation.
        It extends each bound by a multiple of the standard deviation of `x`
        given by `extend_fct`. Defaults to False.
    bound_correction: boolean, optional
        Whether to perform boundary correction on the bounds of `x` or not.
        Defaults to True.
    extend_fct: float, optional
        Number of standard deviations used to widen the
        lower and upper bounds of `x`. Defaults to 0.5.
    bw_fct: float, optional
        A value that multiplies `bw` which enables tuning smoothness by hand.
        Must be positive. Values below 1 decrease smoothness while values
        above 1 decrease it. Defaults to 1 (no modification).
    bw_return: bool, optional
        Whether to return the estimated bandwidth in addition to the
        other objects. Defaults to False.
    custom_lims: list or tuple, optional
        A list or tuple of length 2 indicating custom bounds
        for the range of `x`. Defaults to None which disables custom bounds.
    cumulative: bool, optional
        Whether return the PDF or the cumulative PDF. Defaults to False.
    grid_len: int, optional
        The number of intervals used to bin the data points
        (a.k.a. the length of the grid used in the estimation)
        Defaults to 512.

    Returns
    -------
    grid : Gridded numpy array for the x values.
    pdf : Numpy array for the density estimates.
    bw: optional, the estimated bandwidth.
    """

    # Check `x` is from appropiate type
    x = check_type(x)

    # Assert `bw_fct` is numeric and positive
    assert isinstance(bw_fct, (int, float))
    assert bw_fct > 0

    # Preliminary calculations
    x_len = len(x)
    x_min = x.min()
    x_max = x.max()
    x_std = (((x ** 2).sum() / x_len) - (x.sum() / x_len) ** 2) ** 0.5
    x_range = x_max - x_min

    # Determine grid
    grid_min, grid_max, grid_len = get_grid(
        x_min, x_max, x_std, extend_fct, grid_len,
        custom_lims, extend, bound_correction
    )
    grid_counts, _, grid_edges = histogram(x, grid_len, (grid_min, grid_max))

    # Bandwidth estimation
    bw = bw_fct * get_bw(x, bw, grid_counts, x_std, x_range)

    # Density estimation
    if adaptive:
        grid, pdf = _kde_adaptive(x, bw, grid_edges, grid_counts, grid_len,
                                  bound_correction)
    else:
        grid, pdf = _kde_convolution(x, bw, grid_edges, grid_counts, grid_len,
                                     bound_correction)

    if cumulative:
        pdf = pdf.cumsum() / pdf.sum()

    if bw_return:
        return grid, pdf, bw
    else:
        return grid, pdf


def kde_circular(
    x,
    bw="taylor",
    bw_fct=1,
    bw_return=False,
    custom_lims=None,
    cumulative=False,
    grid_len=512
):
    """
    1 dimensional density estimation for circular data.

    Given an array of data points `x` measured in radians,
    it returns an estimate of the probability density function that generated
    the samples in `x`.

    Parameters
    ----------
    x : 1D numpy array
        Data used to calculate the density estimation.
        Theoritically it is a random sample obtained from $f$,
        the true probability density function we aim to estimate.
    bw: int, float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be
        "taylor" since it is the only option supported so far. Defaults to "taylor".
    bw_fct: float, optional
        A value that multiplies `bw` which enables tuning smoothness by hand.
        Must be positive. Values above 1 decrease smoothness while values
        below 1 decrease it. Defaults to 1 (no modification).
    bw_return: bool, optional
        Whether to return the estimated bandwidth in addition to the
        other objects. Defaults to False.
    custom_lims: list or tuple, optional
        A list or tuple of length 2 indicating custom bounds
        for the range of `x`. Defaults to None which means the estimation
        limits are [-pi, pi].
    cumulative: bool, optional
        Whether return the PDF or the cumulative PDF. Defaults to False.
    grid_len: int, optional
        The number of intervals used to bin the data points
        (a.k.a. the length of the grid used in the estimation)
        Defaults to 512.
    """

    x = check_type(x)

    # All values between -pi and pi
    x[x > np.pi] = x[x > np.pi] - 2 * np.pi
    x[x < -np.pi] = x[x < -np.pi] + 2 * np.pi

    # Assert `bw_fct` is numeric and positive
    assert isinstance(bw_fct, (int, float))
    assert bw_fct > 0

    # Determine bandwidth
    if isinstance(bw, bool):
        raise ValueError((
            "`bw` can't be of type `bool`.\n"
            "Expected a positive numeric or 'taylor'"
        ))
    if isinstance(bw, (int, float)):
        if bw < 0:
            raise ValueError(f"Numeric `bw` must be positive.\nInput: {bw:.4f}.")
    if isinstance(bw, str):
        if bw == "taylor":
            bw = bw_taylor(x)
        else:
            raise ValueError((
                f"`bw` must be a positive numeric or `taylor`, not {bw}"
            ))
    bw *= bw_fct

    # Determine grid
    if custom_lims is not None:
        custom_lims = check_custom_lims(custom_lims, x.min(), x.max())
        grid_min = custom_lims[0]
        grid_max = custom_lims[1]
        assert grid_min >= -np.pi, "Lower limit can't be smaller than -pi"
        assert grid_max <= np.pi, "Upper limit can't be larger than pi"
    else:
        grid_min = -np.pi
        grid_max = np.pi

    bins = np.linspace(grid_min, grid_max, grid_len + 1)
    bin_counts, _, bin_edges = histogram(x, bins=bins)
    grid = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    kern = vonmises_pdf(x=grid, mu=0, kappa=bw)
    pdf = np.fft.fftshift(np.fft.irfft(np.fft.rfft(kern) * np.fft.rfft(bin_counts)))
    pdf /= len(x)

    if cumulative:
        pdf = pdf.cumsum() / pdf.sum()

    if bw_return:
        return grid, pdf, bw
    else:
        return grid, pdf


def _kde_convolution(x, bw, grid_edges, grid_counts, grid_len, bound_correction):
    """
    1 dimensional Gaussian kernel density estimation via
    convolution of the binned relative frequencies and a Gaussian filter.
    This is an internal function used by `kde()`.
    """

    # Calculate relative frequencies per bin
    bin_width = grid_edges[1] - grid_edges[0]
    f = grid_counts / bin_width / len(x)

    # Bandwidth must consider the bin width
    bw /= bin_width

    # See: https://stackoverflow.com/questions/2773606/gaussian-filter-in-matlab
    kernel_n = int(bw * 2 * np.pi)

    # Temporal fix?
    if kernel_n == 0:
        kernel_n = 1
    kernel = gaussian(kernel_n, bw)

    if bound_correction:
        npad = int(grid_len / 5)
        f = np.concatenate([f[npad - 1::-1], f, f[grid_len:grid_len - npad - 1:-1]])
        pdf = convolve(f, kernel, mode="same", method="direct")[npad:npad + grid_len]
        pdf /= bw * (2 * np.pi) ** 0.5
    else:
        pdf = convolve(f, kernel, mode="same", method="direct")
        pdf /= (bw * (2 * np.pi) ** 0.5)

    grid = (grid_edges[1:] + grid_edges[:-1]) / 2
    return grid, pdf


def _kde_adaptive(x, bw, grid_edges, grid_counts, grid_len, bound_correction):
    """
    1 dimensional adaptive Gaussian kernel density estimation.
    The implementation uses the binning technique.
    Since there is not an unique `bw`, the convolution is not possible.
    The alternative implemented in this function is known as Abramson's method.
    This is an internal function used by `kde()`.
    """
    # Pilot computations used for bandwidth adjustment
    pilot_grid, pilot_pdf = _kde_convolution(x, bw, grid_edges, grid_counts,
                                             grid_len, bound_correction)

    # Adds to avoid np.log(0) and zero division
    pilot_pdf += 1e-9

    # Determine the modification factors
    pdf_interp = np.interp(x, pilot_grid, pilot_pdf)
    geom_mean = np.exp(np.mean(np.log(pdf_interp)))

    # Power of c = 0.5 -> Abramson's method
    adj_factor = (geom_mean / pilot_pdf) ** 0.5
    bw_adj = bw * adj_factor

    # Estimation of Gaussian KDE via binned method (convolution not possible)
    grid = pilot_grid

    if bound_correction:
        grid_npad = int(grid_len / 5)
        grid_width = grid_edges[1] - grid_edges[0]
        grid_pad = grid_npad * grid_width
        grid_padded = np.linspace(
            grid_edges[0] - grid_pad,
            grid_edges[grid_len - 1] + grid_pad,
            num = grid_len + 2 * grid_npad
        )
        grid_counts = np.concatenate([
            grid_counts[grid_npad - 1:: -1],
            grid_counts,
            grid_counts[grid_len:grid_len - grid_npad - 1: -1]]
        )
        bw_adj = np.concatenate([
            bw_adj[grid_npad - 1:: -1],
            bw_adj,
            bw_adj[grid_len:grid_len - grid_npad - 1: -1]]
        )
        pdf_mat = ((grid_padded - grid_padded[:, None]) / bw_adj[:, None])
        pdf_mat = np.exp(-0.5 * pdf_mat ** 2) * grid_counts[:, None]
        pdf_mat /= ((2 * np.pi) ** 0.5 * bw_adj[:, None])
        pdf = np.sum(pdf_mat[:, grid_npad:grid_npad + grid_len], axis=0) / len(x)

    else:
        pdf_mat = ((grid - grid[:, None]) / bw_adj[:, None])
        pdf_mat = np.exp(-0.5 * pdf_mat ** 2) * grid_counts[:, None]
        pdf_mat /= ((2 * np.pi) ** 0.5 * bw_adj[:, None])
        pdf = np.sum(pdf_mat, axis=0) / len(x)

    return grid, pdf
