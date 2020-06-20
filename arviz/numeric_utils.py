"""Numerical utility functions for ArviZ."""
import warnings
import numpy as np
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix

from .stats.stats_utils import histogram
from .utils import _stack, _dot, _cov


def _fast_kde(x, cumulative=False, bw=4.5, xmin=None, xmax=None):
    """Fast Fourier transform-based Gaussian kernel density estimate (KDE).

    The code was adapted from https://github.com/mfouesneau/faststats

    Parameters
    ----------
    x : Numpy array or list
    cumulative : bool
        If true, estimate the cdf instead of the pdf
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy).
    xmin : float
        Manually set lower limit.
    xmax : float
        Manually set upper limit.

    Returns
    -------
    density: A gridded 1D KDE of the input points (x)
    xmin: minimum value of x
    xmax: maximum value of x
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        warnings.warn("kde plot failed, you may want to check your data")
        return np.array([np.nan]), np.nan, np.nan

    len_x = len(x)
    n_points = 200 if (xmin or xmax) is None else 500

    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)

    assert np.min(x) >= xmin
    assert np.max(x) <= xmax

    log_len_x = np.log(len_x) * bw

    n_bins = min(int(len_x ** (1 / 3) * log_len_x * 2), n_points)
    if n_bins < 2:
        warnings.warn("kde plot failed, you may want to check your data")
        return np.array([np.nan]), np.nan, np.nan

    # hist, bin_edges = np.histogram(x, bins=n_bins, range=(xmin, xmax))
    # grid = hist / (hist.sum() * np.diff(bin_edges))

    _, grid, _ = histogram(x, n_bins, range_hist=(xmin, xmax))

    scotts_factor = len_x ** (-0.2)
    kern_nx = int(scotts_factor * 2 * np.pi * log_len_x)
    kernel = gaussian(kern_nx, scotts_factor * log_len_x)

    npad = min(n_bins, 2 * kern_nx)
    grid = np.concatenate([grid[npad:0:-1], grid, grid[n_bins : n_bins - npad : -1]])
    density = convolve(grid, kernel, mode="same", method="direct")[npad : npad + n_bins]
    norm_factor = (2 * np.pi * log_len_x ** 2 * scotts_factor ** 2) ** 0.5

    density /= norm_factor

    if cumulative:
        density = density.cumsum() / density.sum()

    return density, xmin, xmax


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
        If True, use circular boundaries. Defaults to False
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
    std_devs = np.diag(cov ** 0.5)
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    inv_cov = np.linalg.inv(cov * scotts_factor ** 2)

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

    norm_factor = np.linalg.det(2 * np.pi * cov * scotts_factor ** 2)
    norm_factor = len_x * d_x * d_y * norm_factor ** 0.5

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
    Computes the width of the bins by taking the maximun of the Sturges and the Freedman-Diaconis
    estimators. Acording to numpy `np.histogram` this provides good all around performance.

    The Sturges is a very simplistic estimator based on the assumption of normality of the data.
    This estimator has poor performance for non-normal data, which becomes especially obvious for
    large data sets. The estimate depends only on size of the data.

    The Freedman-Diaconis rule uses interquartile range (IQR) to estimate the binwidth.
    It is considered a robusts version of the Scott rule as the IQR is less affected by outliers
    than the standard deviation. However, the IQR depends on fewer points than the standard
    deviation, so it is less accurate, especially for long tailed distributions.
    """
    x_min = values.min().astype(int)
    x_max = values.max().astype(int)

    # Sturges histogram bin estimator
    bins_sturges = (x_max - x_min) / (np.log2(values.size) + 1)

    # The Freedman-Diaconis histogram bin estimator.
    iqr = np.subtract(*np.percentile(values, [75, 25]))  # pylint: disable=assignment-from-no-return
    bins_fd = 2 * iqr * values.size ** (-1 / 3)

    width = np.round(np.max([1, bins_sturges, bins_fd])).astype(int)

    return np.arange(x_min, x_max + width + 1, width)


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
