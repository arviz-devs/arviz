# pylint: disable=invalid-name,too-many-lines
"""Density estimation functions for ArviZ."""
import warnings

import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d, gaussian  # pylint: disable=no-name-in-module
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module

from ..utils import _cov, _dot, _stack, conditional_jit

__all__ = ["kde"]


def _bw_scott(x, x_std=None, **kwargs):  # pylint: disable=unused-argument
    """Scott's Rule."""
    if x_std is None:
        x_std = np.std(x)
    bw = 1.06 * x_std * len(x) ** (-0.2)
    return bw


def _bw_silverman(x, x_std=None, **kwargs):  # pylint: disable=unused-argument
    """Silverman's Rule."""
    if x_std is None:
        x_std = np.std(x)
    q75, q25 = np.percentile(x, [75, 25])
    x_iqr = q75 - q25
    a = min(x_std, x_iqr / 1.34)
    bw = 0.9 * a * len(x) ** (-0.2)
    return bw


def _bw_isj(x, grid_counts=None, x_std=None, x_range=None):
    """Improved Sheather-Jones bandwidth estimation.

    Improved Sheather and Jones method as explained in [1]_. This method is used internally by the
    KDE estimator, resulting in saved computation time as minimums, maximums and the grid are
    pre-computed.

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
    a_k = _dct1d(grid_relfreq)

    k_sq = np.arange(1, grid_len) ** 2
    a_sq = a_k[range(1, grid_len)] ** 2

    t = _root(_fixed_point, x_len, args=(x_len, k_sq, a_sq), x=x)
    h = t**0.5 * x_range
    return h


def _bw_experimental(x, grid_counts=None, x_std=None, x_range=None):
    """Experimental bandwidth estimator."""
    bw_silverman = _bw_silverman(x, x_std=x_std)
    bw_isj = _bw_isj(x, grid_counts=grid_counts, x_range=x_range)
    return 0.5 * (bw_silverman + bw_isj)


def _bw_taylor(x):
    """Taylor's rule for circular bandwidth estimation.

    This function implements a rule-of-thumb for choosing the bandwidth of a von Mises kernel
    density estimator that assumes the underlying distribution is von Mises as introduced in [1]_.
    It is analogous to Scott's rule for the Gaussian KDE.

    Circular bandwidth has a different scale from linear bandwidth. Unlike linear scale, low
    bandwidths are associated with oversmoothing and high values with undersmoothing.

    References
    ----------
    .. [1] C.C Taylor (2008). Automatic bandwidth selection for circular
           density estimation.
           Computational Statistics and Data Analysis, 52, 7, 3493–3500.
    """
    x_len = len(x)
    kappa = _kappa_mle(x)
    num = 3 * x_len * kappa**2 * ive(2, 2 * kappa)
    den = 4 * np.pi**0.5 * ive(0, kappa) ** 2
    return (num / den) ** 0.4


_BW_METHODS_LINEAR = {
    "scott": _bw_scott,
    "silverman": _bw_silverman,
    "isj": _bw_isj,
    "experimental": _bw_experimental,
}


def _get_bw(x, bw, grid_counts=None, x_std=None, x_range=None):
    """Compute bandwidth for a given data `x` and `bw`.

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
        raise ValueError(
            (
                "`bw` must not be of type `bool`.\n"
                "Expected a positive numeric or one of the following strings:\n"
                f"{list(_BW_METHODS_LINEAR)}."
            )
        )
    if isinstance(bw, (int, float)):
        if bw < 0:
            raise ValueError(f"Numeric `bw` must be positive.\nInput: {bw:.4f}.")
    elif isinstance(bw, str):
        bw_lower = bw.lower()

        if bw_lower not in _BW_METHODS_LINEAR:
            raise ValueError(
                "Unrecognized bandwidth method.\n"
                f"Input is: {bw_lower}.\n"
                f"Expected one of: {list(_BW_METHODS_LINEAR)}."
            )

        bw_fun = _BW_METHODS_LINEAR[bw_lower]
        bw = bw_fun(x, grid_counts=grid_counts, x_std=x_std, x_range=x_range)
    else:
        raise ValueError(
            "Unrecognized `bw` argument.\n"
            "Expected a positive numeric or one of the following strings:\n"
            f"{list(_BW_METHODS_LINEAR)}."
        )
    return bw


def _vonmises_pdf(x, mu, kappa):
    """Calculate vonmises_pdf."""
    if kappa <= 0:
        raise ValueError("Argument 'kappa' must be positive.")
    pdf = 1 / (2 * np.pi * ive(0, kappa)) * np.exp(np.cos(x - mu) - 1) ** kappa
    return pdf


def _a1inv(x):
    """Compute inverse function.

    Inverse function of the ratio of the first and
    zeroth order Bessel functions of the first kind.

    Returns the value k, such that a1inv(x) = k, i.e. a1(k) = x.
    """
    if 0 <= x < 0.53:
        return 2 * x + x**3 + (5 * x**5) / 6
    elif x < 0.85:
        return -0.4 + 1.39 * x + 0.43 / (1 - x)
    else:
        return 1 / (x**3 - 4 * x**2 + 3 * x)


def _kappa_mle(x):
    mean = _circular_mean(x)
    kappa = _a1inv(np.mean(np.cos(x - mean)))
    return kappa


def _dct1d(x):
    """Discrete Cosine Transform in 1 Dimension.

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

    w_1k = np.r_[1, (2 * np.exp(-(0 + 1j) * (np.arange(1, x_len)) * np.pi / (2 * x_len)))]
    output = np.real(w_1k * fft(x))

    return output


def _fixed_point(t, N, k_sq, a_sq):
    """Calculate t-zeta*gamma^[l](t).

    Implementation of the function t-zeta*gamma^[l](t) derived from equation (30) in [1].

    References
    ----------
    .. [1] Kernel density estimation via diffusion.
       Z. I. Botev, J. F. Grotowski, and D. P. Kroese.
       Ann. Statist. 38 (2010), no. 5, 2916--2957.
    """
    k_sq = np.asfarray(k_sq, dtype=np.float64)
    a_sq = np.asfarray(a_sq, dtype=np.float64)

    l = 7
    f = np.sum(np.power(k_sq, l) * a_sq * np.exp(-k_sq * np.pi**2 * t))
    f *= 0.5 * np.pi ** (2.0 * l)

    for j in np.arange(l - 1, 2 - 1, -1):
        c1 = (1 + 0.5 ** (j + 0.5)) / 3
        c2 = np.product(np.arange(1.0, 2 * j + 1, 2, dtype=np.float64))
        c2 /= (np.pi / 2) ** 0.5
        t_j = np.power((c1 * (c2 / (N * f))), (2.0 / (3.0 + 2.0 * j)))
        f = np.sum(k_sq**j * a_sq * np.exp(-k_sq * np.pi**2.0 * t_j))
        f *= 0.5 * np.pi ** (2 * j)

    out = t - (2 * N * np.pi**0.5 * f) ** (-0.4)
    return out


def _root(function, N, args, x):
    # The right bound is at most 0.01
    found = False
    N = max(min(1050, N), 50)
    tol = 10e-12 + 0.01 * (N - 50) / 1000

    while not found:
        try:
            bw, res = brentq(function, 0, 0.01, args=args, full_output=True, disp=False)
            found = res.converged
        except ValueError:
            bw = 0
            tol *= 2.0
            found = False
        if bw <= 0 or tol >= 1:
            bw = (_bw_silverman(x) / np.ptp(x)) ** 2
            return bw
    return bw


def _check_custom_lims(custom_lims, x_min, x_max):
    """Check if `custom_lims` are of the correct type.

    It accepts numeric lists/tuples of length 2.

    Parameters
    ----------
    custom_lims : Object whose type is checked.

    Returns
    -------
    None: Object of type None
    """
    if not isinstance(custom_lims, (list, tuple)):
        raise TypeError(
            "`custom_lims` must be a numeric list or tuple of length 2.\n"
            f"Not an object of {type(custom_lims)}."
        )

    if len(custom_lims) != 2:
        raise AttributeError(f"`len(custom_lims)` must be 2, not {len(custom_lims)}.")

    any_bool = any(isinstance(i, bool) for i in custom_lims)
    if any_bool:
        raise TypeError("Elements of `custom_lims` must be numeric or None, not bool.")

    custom_lims = list(custom_lims)  # convert to a mutable object
    if custom_lims[0] is None:
        custom_lims[0] = x_min

    if custom_lims[1] is None:
        custom_lims[1] = x_max

    all_numeric = all(isinstance(i, (int, float, np.integer, np.float)) for i in custom_lims)
    if not all_numeric:
        raise TypeError(
            ("Elements of `custom_lims` must be numeric or None.\n" "At least one of them is not.")
        )

    if not custom_lims[0] < custom_lims[1]:
        raise ValueError("`custom_lims[0]` must be smaller than `custom_lims[1]`.")

    if custom_lims[0] > x_min or custom_lims[1] < x_max:
        raise ValueError("Some observations are outside `custom_lims` boundaries.")

    return custom_lims


def _get_grid(
    x_min, x_max, x_std, extend_fct, grid_len, custom_lims, extend=True, bound_correction=False
):
    """Compute the grid that bins the data used to estimate the density function.

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
        Must be numeric of length 2. Overrides `extend`.
    extend: bool, optional
        Whether to extend the range of the data or not.
        Default is True.
    bound_correction: bool, optional
        Whether the density estimations performs boundary correction or not.
        This does not impacts directly in the output, but is used
        to override `extend`. Overrides `extend`.
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
    grid_len = max(int(grid_len), 100)

    # Set up domain
    if custom_lims is not None:
        custom_lims = _check_custom_lims(custom_lims, x_min, x_max)
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


def kde(x, circular=False, **kwargs):
    """One dimensional density estimation.

    It is a wrapper around ``kde_linear()`` and ``kde_circular()``.

    Parameters
    ----------
    x: 1D numpy array
        Data used to calculate the density estimation.
    circular: bool, optional
        Whether ``x`` is a circular variable or not. Defaults to False.
    **kwargs
        Arguments passed to ``kde_linear()`` and ``kde_circular()``.
        See their documentation for more info.

    Returns
    -------
    grid: Gridded numpy array for the x values.
    pdf: Numpy array for the density estimates.
    bw: optional, the estimated bandwidth.

    Examples
    --------
    Default density estimation for linear data

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from arviz import kde
        >>>
        >>> rvs = np.random.gamma(shape=1.8, size=1000)
        >>> grid, pdf = kde(rvs)
        >>> plt.plot(grid, pdf)
        >>> plt.show()

    Density estimation for linear data with Silverman's rule bandwidth

    .. plot::
        :context: close-figs

        >>> grid, pdf = kde(rvs, bw="silverman")
        >>> plt.plot(grid, pdf)
        >>> plt.show()

    Density estimation for linear data with scaled bandwidth

    .. plot::
        :context: close-figs

        >>> # bw_fct > 1 means more smoothness.
        >>> grid, pdf = kde(rvs, bw_fct=2.5)
        >>> plt.plot(grid, pdf)
        >>> plt.show()

    Default density estimation for linear data with extended limits

    .. plot::
        :context: close-figs

        >>> grid, pdf = kde(rvs, bound_correction=False, extend=True, extend_fct=0.5)
        >>> plt.plot(grid, pdf)
        >>> plt.show()

    Default density estimation for linear data with custom limits

    .. plot::
        :context: close-figs

        >>> # It accepts tuples and lists of length 2.
        >>> grid, pdf = kde(rvs, bound_correction=False, custom_lims=(0, 10))
        >>> plt.plot(grid, pdf)
        >>> plt.show()

    Default density estimation for circular data

    .. plot::
        :context: close-figs

        >>> rvs = np.random.vonmises(mu=np.pi, kappa=1, size=500)
        >>> grid, pdf = kde(rvs, circular=True)
        >>> plt.plot(grid, pdf)
        >>> plt.show()

    Density estimation for circular data with scaled bandwidth

    .. plot::
        :context: close-figs

        >>> rvs = np.random.vonmises(mu=np.pi, kappa=1, size=500)
        >>> # bw_fct > 1 means less smoothness.
        >>> grid, pdf = kde(rvs, circular=True, bw_fct=3)
        >>> plt.plot(grid, pdf)
        >>> plt.show()

    Density estimation for circular data with custom limits

    .. plot::
        :context: close-figs

        >>> # This is still experimental, does not always work.
        >>> rvs = np.random.vonmises(mu=0, kappa=30, size=500)
        >>> grid, pdf = kde(rvs, circular=True, custom_lims=(-1, 1))
        >>> plt.plot(grid, pdf)
        >>> plt.show()

    See Also
    --------
    plot_kde : Compute and plot a kernel density estimate.
    """
    x = x[np.isfinite(x)]
    if x.size == 0 or np.all(x == x[0]):
        warnings.warn("Your data appears to have a single value or no finite values")

        return np.zeros(2), np.array([np.nan] * 2)

    if circular:
        if circular == "degrees":
            x = np.radians(x)
        kde_fun = _kde_circular
    else:
        kde_fun = _kde_linear

    return kde_fun(x, **kwargs)


def _kde_linear(
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
    grid_len=512,
    **kwargs,  # pylint: disable=unused-argument
):
    """One dimensional density estimation for linear data.

    Given an array of data points `x` it returns an estimate of
    the probability density function that generated the samples in `x`.

    Parameters
    ----------
    x : 1D numpy array
        Data used to calculate the density estimation.
    bw: int, float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be one of "scott",
        "silverman", "isj" or "experimental". Defaults to "experimental".
    adaptive: boolean, optional
        Indicates if the bandwidth is adaptive or not.
        It is the recommended approach when there are multiple modes with different spread.
        It is not compatible with convolution. Defaults to False.
    extend: boolean, optional
        Whether to extend the observed range for `x` in the estimation.
        It extends each bound by a multiple of the standard deviation of `x` given by `extend_fct`.
        Defaults to False.
    bound_correction: boolean, optional
        Whether to perform boundary correction on the bounds of `x` or not.
        Defaults to True.
    extend_fct: float, optional
        Number of standard deviations used to widen the lower and upper bounds of `x`.
        Defaults to 0.5.
    bw_fct: float, optional
        A value that multiplies `bw` which enables tuning smoothness by hand.
        Must be positive. Values below 1 decrease smoothness while values above 1 decrease it.
        Defaults to 1 (no modification).
    bw_return: bool, optional
        Whether to return the estimated bandwidth in addition to the other objects.
        Defaults to False.
    custom_lims: list or tuple, optional
        A list or tuple of length 2 indicating custom bounds for the range of `x`.
        Defaults to None which disables custom bounds.
    cumulative: bool, optional
        Whether return the PDF or the cumulative PDF. Defaults to False.
    grid_len: int, optional
        The number of intervals used to bin the data points i.e. the length of the grid used in
        the estimation. Defaults to 512.

    Returns
    -------
    grid : Gridded numpy array for the x values.
    pdf : Numpy array for the density estimates.
    bw: optional, the estimated bandwidth.
    """
    # Check `bw_fct` is numeric and positive
    if not isinstance(bw_fct, (int, float, np.integer, np.floating)):
        raise TypeError(f"`bw_fct` must be a positive number, not an object of {type(bw_fct)}.")

    if bw_fct <= 0:
        raise ValueError(f"`bw_fct` must be a positive number, not {bw_fct}.")

    # Preliminary calculations
    x_min = x.min()
    x_max = x.max()
    x_std = np.std(x)
    x_range = x_max - x_min

    # Determine grid
    grid_min, grid_max, grid_len = _get_grid(
        x_min, x_max, x_std, extend_fct, grid_len, custom_lims, extend, bound_correction
    )
    grid_counts, _, grid_edges = histogram(x, grid_len, (grid_min, grid_max))

    # Bandwidth estimation
    bw = bw_fct * _get_bw(x, bw, grid_counts, x_std, x_range)

    # Density estimation
    if adaptive:
        grid, pdf = _kde_adaptive(x, bw, grid_edges, grid_counts, grid_len, bound_correction)
    else:
        grid, pdf = _kde_convolution(x, bw, grid_edges, grid_counts, grid_len, bound_correction)

    if cumulative:
        pdf = pdf.cumsum() / pdf.sum()

    if bw_return:
        return grid, pdf, bw
    else:
        return grid, pdf


def _kde_circular(
    x,
    bw="taylor",
    bw_fct=1,
    bw_return=False,
    custom_lims=None,
    cumulative=False,
    grid_len=512,
    **kwargs,  # pylint: disable=unused-argument
):
    """One dimensional density estimation for circular data.

    Given an array of data points `x` measured in radians, it returns an estimate of the
    probability density function that generated the samples in `x`.

    Parameters
    ----------
    x : 1D numpy array
        Data used to calculate the density estimation.
    bw: int, float or str, optional
        If numeric, indicates the bandwidth and must be positive.
        If str, indicates the method to estimate the bandwidth and must be "taylor" since it is the
        only option supported so far. Defaults to "taylor".
    bw_fct: float, optional
        A value that multiplies `bw` which enables tuning smoothness by hand. Must be positive.
        Values above 1 decrease smoothness while values below 1 decrease it.
        Defaults to 1 (no modification).
    bw_return: bool, optional
        Whether to return the estimated bandwidth in addition to the other objects.
        Defaults to False.
    custom_lims: list or tuple, optional
        A list or tuple of length 2 indicating custom bounds for the range of `x`.
        Defaults to None which means the estimation limits are [-pi, pi].
    cumulative: bool, optional
        Whether return the PDF or the cumulative PDF. Defaults to False.
    grid_len: int, optional
        The number of intervals used to bin the data pointa i.e. the length of the grid used in the
        estimation. Defaults to 512.
    """
    # All values between -pi and pi
    x = _normalize_angle(x)

    # Check `bw_fct` is numeric and positive
    if not isinstance(bw_fct, (int, float, np.integer, np.floating)):
        raise TypeError(f"`bw_fct` must be a positive number, not an object of {type(bw_fct)}.")

    if bw_fct <= 0:
        raise ValueError(f"`bw_fct` must be a positive number, not {bw_fct}.")

    # Determine bandwidth
    if isinstance(bw, bool):
        raise ValueError(
            "`bw` can't be of type `bool`.\n" "Expected a positive numeric or 'taylor'"
        )
    if isinstance(bw, (int, float)):
        if bw < 0:
            raise ValueError(f"Numeric `bw` must be positive.\nInput: {bw:.4f}.")
    if isinstance(bw, str):
        if bw == "taylor":
            bw = _bw_taylor(x)
        else:
            raise ValueError(f"`bw` must be a positive numeric or `taylor`, not {bw}")
    bw *= bw_fct

    # Determine grid
    if custom_lims is not None:
        custom_lims = _check_custom_lims(custom_lims, x.min(), x.max())
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

    kern = _vonmises_pdf(x=grid, mu=0, kappa=bw)
    pdf = np.fft.fftshift(np.fft.irfft(np.fft.rfft(kern) * np.fft.rfft(bin_counts)))
    pdf /= len(x)

    if cumulative:
        pdf = pdf.cumsum() / pdf.sum()

    if bw_return:
        return grid, pdf, bw
    else:
        return grid, pdf


# pylint: disable=unused-argument
def _kde_convolution(x, bw, grid_edges, grid_counts, grid_len, bound_correction, **kwargs):
    """Kernel density with convolution.

    One dimensional Gaussian kernel density estimation via convolution of the binned relative
    frequencies and a Gaussian filter. This is an internal function used by `kde()`.
    """
    # Calculate relative frequencies per bin
    bin_width = grid_edges[1] - grid_edges[0]
    f = grid_counts / bin_width / len(x)

    # Bandwidth must consider the bin width
    bw /= bin_width

    # See: https://stackoverflow.com/questions/2773606/gaussian-filter-in-matlab

    grid = (grid_edges[1:] + grid_edges[:-1]) / 2

    kernel_n = int(bw * 2 * np.pi)
    if kernel_n == 0:
        kernel_n = 1

    kernel = gaussian(kernel_n, bw)

    if bound_correction:
        npad = int(grid_len / 5)
        f = np.concatenate([f[npad - 1 :: -1], f, f[grid_len : grid_len - npad - 1 : -1]])
        pdf = convolve(f, kernel, mode="same", method="direct")[npad : npad + grid_len]
        pdf /= bw * (2 * np.pi) ** 0.5
    else:
        pdf = convolve(f, kernel, mode="same", method="direct")
        pdf /= bw * (2 * np.pi) ** 0.5

    return grid, pdf


def _kde_adaptive(x, bw, grid_edges, grid_counts, grid_len, bound_correction, **kwargs):
    """Compute Adaptive Kernel Density Estimation.

    One dimensional adaptive Gaussian kernel density estimation. The implementation uses the binning
    technique. Since there is not an unique `bw`, the convolution is not possible. The alternative
    implemented in this function is known as Abramson's method.
    This is an internal function used by `kde()`.
    """
    # Pilot computations used for bandwidth adjustment
    pilot_grid, pilot_pdf = _kde_convolution(
        x, bw, grid_edges, grid_counts, grid_len, bound_correction
    )

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
            num=grid_len + 2 * grid_npad,
        )
        grid_counts = np.concatenate(
            [
                grid_counts[grid_npad - 1 :: -1],
                grid_counts,
                grid_counts[grid_len : grid_len - grid_npad - 1 : -1],
            ]
        )
        bw_adj = np.concatenate(
            [bw_adj[grid_npad - 1 :: -1], bw_adj, bw_adj[grid_len : grid_len - grid_npad - 1 : -1]]
        )
        pdf_mat = (grid_padded - grid_padded[:, None]) / bw_adj[:, None]
        pdf_mat = np.exp(-0.5 * pdf_mat**2) * grid_counts[:, None]
        pdf_mat /= (2 * np.pi) ** 0.5 * bw_adj[:, None]
        pdf = np.sum(pdf_mat[:, grid_npad : grid_npad + grid_len], axis=0) / len(x)

    else:
        pdf_mat = (grid - grid[:, None]) / bw_adj[:, None]
        pdf_mat = np.exp(-0.5 * pdf_mat**2) * grid_counts[:, None]
        pdf_mat /= (2 * np.pi) ** 0.5 * bw_adj[:, None]
        pdf = np.sum(pdf_mat, axis=0) / len(x)

    return grid, pdf


def _fast_kde_2d(x, y, gridsize=(128, 128), circular=False):
    """
    2D fft-based Gaussian kernel density estimate (KDE).

    The code was adapted from https://github.com/mfouesneau/faststats

    Parameters
    ----------
    x : Numpy array or list
    y : Numpy array or list
    gridsize : tuple
        Number of points used to discretize data. Use powers of 2 for fft optimization
    circular: bool
        If True use circular boundaries. Defaults to False

    Returns
    -------
    grid: A gridded 2D KDE of the input points (x, y)
    xmin: minimum value of x
    xmax: maximum value of x
    ymin: minimum value of y
    ymax: maximum value of y
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    len_x = len(x)
    weights = np.ones(len_x)
    n_x, n_y = gridsize

    d_x = (xmax - xmin) / (n_x - 1)
    d_y = (ymax - ymin) / (n_y - 1)

    xyi = _stack(x, y).T
    xyi -= [xmin, ymin]
    xyi /= [d_x, d_y]
    xyi = np.floor(xyi, xyi).T

    scotts_factor = len_x ** (-1 / 6)
    cov = _cov(xyi)
    std_devs = np.diag(cov) ** 0.5
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    inv_cov = np.linalg.inv(cov * scotts_factor**2)

    x_x = np.arange(kern_nx) - kern_nx / 2
    y_y = np.arange(kern_ny) - kern_ny / 2
    x_x, y_y = np.meshgrid(x_x, y_y)

    kernel = _stack(x_x.flatten(), y_y.flatten())
    kernel = _dot(inv_cov, kernel) * kernel
    kernel = np.exp(-kernel.sum(axis=0) / 2)
    kernel = kernel.reshape((int(kern_ny), int(kern_nx)))

    boundary = "wrap" if circular else "symm"

    grid = coo_matrix((weights, xyi), shape=(n_x, n_y)).toarray()
    grid = convolve2d(grid, kernel, mode="same", boundary=boundary)

    norm_factor = np.linalg.det(2 * np.pi * cov * scotts_factor**2)
    norm_factor = len_x * d_x * d_y * norm_factor**0.5

    grid /= norm_factor

    return grid, xmin, xmax, ymin, ymax


def get_bins(values):
    """
    Automatically compute the number of bins for discrete variables.

    Parameters
    ----------
    values = numpy array
        values

    Returns
    -------
    array with the bins

    Notes
    -----
    Computes the width of the bins by taking the maximum of the Sturges and the Freedman-Diaconis
    estimators. According to numpy `np.histogram` this provides good all around performance.

    The Sturges is a very simplistic estimator based on the assumption of normality of the data.
    This estimator has poor performance for non-normal data, which becomes especially obvious for
    large data sets. The estimate depends only on size of the data.

    The Freedman-Diaconis rule uses interquartile range (IQR) to estimate the binwidth.
    It is considered a robust version of the Scott rule as the IQR is less affected by outliers
    than the standard deviation. However, the IQR depends on fewer points than the standard
    deviation, so it is less accurate, especially for long tailed distributions.
    """
    dtype = values.dtype.kind

    if dtype == "i":
        x_min = values.min().astype(int)
        x_max = values.max().astype(int)
    else:
        x_min = values.min().astype(float)
        x_max = values.max().astype(float)

    # Sturges histogram bin estimator
    bins_sturges = (x_max - x_min) / (np.log2(values.size) + 1)

    # The Freedman-Diaconis histogram bin estimator.
    iqr = np.subtract(*np.percentile(values, [75, 25]))  # pylint: disable=assignment-from-no-return
    bins_fd = 2 * iqr * values.size ** (-1 / 3)

    if dtype == "i":
        width = np.round(np.max([1, bins_sturges, bins_fd])).astype(int)
        bins = np.arange(x_min, x_max + width + 1, width)
    else:
        width = np.max([bins_sturges, bins_fd])
        if np.isclose(x_min, x_max):
            width = 1e-3
        bins = np.arange(x_min, x_max + width, width)

    return bins


def _sturges_formula(dataset, mult=1):
    """Use Sturges' formula to determine number of bins.

    See https://en.wikipedia.org/wiki/Histogram#Sturges'_formula
    or https://doi.org/10.1080%2F01621459.1926.10502161

    Parameters
    ----------
    dataset: xarray.DataSet
       Must have the `draw` dimension

    mult: float
        Used to scale the number of bins up or down. Default is 1 for Sturges' formula.

    Returns
    -------
    int
        Number of bins to use
    """
    return int(np.ceil(mult * np.log2(dataset.draw.size)) + 1)


def _circular_mean(x):
    """Compute mean of circular variable measured in radians.

    The result is between -pi and pi.
    """
    sinr = np.sum(np.sin(x))
    cosr = np.sum(np.cos(x))
    mean = np.arctan2(sinr, cosr)

    return mean


def _normalize_angle(x, zero_centered=True):
    """Normalize angles.

    Normalize angles in radians to [-pi, pi) or [0, 2 * pi) according to `zero_centered`.
    """
    if zero_centered:
        return (x + np.pi) % (2 * np.pi) - np.pi
    else:
        return x % (2 * np.pi)


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


def _find_hdi_contours(density, hdi_probs):
    """
    Find contours enclosing regions of highest posterior density.

    Parameters
    ----------
    density : array-like
        A 2D KDE on a grid with cells of equal area.
    hdi_probs : array-like
        An array of highest density interval confidence probabilities.

    Returns
    -------
    contour_levels : array
        The contour levels corresponding to the given HDI probabilities.
    """
    # Using the algorithm from corner.py
    sorted_density = np.sort(density, axis=None)[::-1]
    sm = sorted_density.cumsum()
    sm /= sm[-1]

    contours = np.empty_like(hdi_probs)
    for idx, hdi_prob in enumerate(hdi_probs):
        try:
            contours[idx] = sorted_density[sm <= hdi_prob][-1]
        except IndexError:
            contours[idx] = sorted_density[0]

    return contours
