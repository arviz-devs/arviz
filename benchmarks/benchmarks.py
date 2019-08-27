# Write the benchmarking functions here.
# See "Writing benchmarks" in the airspeed velocity docs for more information.
# https://asv.readthedocs.io/en/stable/
import numpy as np
from numpy import newaxis
from scipy.stats import circstd
from scipy.sparse import coo_matrix
import scipy.signal as ss
import warnings


class Hist:
    def time_histogram(self):
        try:
            data = np.random.rand(10000, 1000)
            import numba

            @numba.njit(cache=True)
            def _hist(data):
                return np.histogram(data, bins=100)

            return _hist(data)
        except ImportError:
            data = np.random.rand(10000, 1000)
            return np.histogram(data, bins=100)


class Variance:
    def time_variance(self):
        try:
            data = np.random.randn(10000, 10000)
            import numba

            @numba.njit(cache=True)
            def stats_variance_1d(data, ddof=0):
                a, b = 0, 0
                for i in data:
                    a = a + i
                    b = b + i * i
                var = b / (len(data)) - ((a / (len(data))) ** 2)
                var = var * (len(data) / (len(data) - ddof))
                return var

            def stats_variance_2d(data, ddof=0, axis=1):
                a, b = data.shape
                if axis == 1:
                    var = np.zeros(a)
                    for i in range(a):
                        var[i] = stats_variance_1d(data[i], ddof=ddof)
                else:
                    var = np.zeros(b)
                    for i in range(b):
                        var[i] = stats_variance_1d(data[:, i], ddof=ddof)
                return var

            return stats_variance_2d(data)
        except ImportError:
            data = np.random.randn(10000, 10000)
            return np.var(data, axis=1)


class CircStd:
    def time_circ_std(self):
        try:
            data = np.random.randn(10000, 1000)
            import numba

            def _circfunc(samples, high, low):
                samples = np.asarray(samples)
                if samples.size == 0:
                    return np.nan, np.nan
                return samples, _angle(samples, low, high, np.pi)

            @numba.vectorize(nopython=True)
            def _angle(samples, low, high, pi=np.pi):
                ang = (samples - low) * 2.0 * pi / (high - low)
                return ang

            def _circular_standard_deviation(samples, high=2 * np.pi, low=0, axis=None):
                pi = np.pi
                samples, ang = _circfunc(samples, high, low)
                S = np.sin(ang).mean(axis=axis)
                C = np.cos(ang).mean(axis=axis)
                R = np.hypot(S, C)
                return ((high - low) / 2.0 / pi) * np.sqrt(-2 * np.log(R))

            return _circular_standard_deviation(data)
        except ImportError:
            data = np.random.randn(10000, 1000)
            return circstd(data)


class Fast_Kde_1d:
    def time_fast_kde(self):

        try:
            x = np.random.randn(10000, 100)
            import numba

            def _fast_kde(x, cumulative=False, bw=4.5, xmin=None, xmax=None):
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

                d_x = (xmax - xmin) / (n_bins - 1)
                grid = _histogram(x, n_bins, range_hist=(xmin, xmax))

                scotts_factor = len_x ** (-0.2)
                kern_nx = int(scotts_factor * 2 * np.pi * log_len_x)
                kernel = ss.gaussian(kern_nx, scotts_factor * log_len_x)

                npad = min(n_bins, 2 * kern_nx)
                grid = np.concatenate([grid[npad:0:-1], grid, grid[n_bins : n_bins - npad : -1]])
                density = ss.convolve(grid, kernel, mode="same", method="direct")[
                    npad : npad + n_bins
                ]
                norm_factor = len_x * d_x * (2 * np.pi * log_len_x ** 2 * scotts_factor ** 2) ** 0.5

                density /= norm_factor

                if cumulative:
                    density = density.cumsum() / density.sum()

                return density, xmin, xmax

            @numba.njit(cache=True)
            def _histogram(x, n_bins, range_hist=None):
                grid, _ = np.histogram(x, bins=n_bins, range=range_hist)
                return grid

            return _fast_kde(x)

        except ImportError:

            x = np.random.randn(10000, 100)

            def _fast_kde(x, cumulative=False, bw=4.5, xmin=None, xmax=None):
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

                d_x = (xmax - xmin) / (n_bins - 1)
                grid = _histogram(x, n_bins, range_hist=(xmin, xmax))

                scotts_factor = len_x ** (-0.2)
                kern_nx = int(scotts_factor * 2 * np.pi * log_len_x)
                kernel = ss.gaussian(kern_nx, scotts_factor * log_len_x)

                npad = min(n_bins, 2 * kern_nx)
                grid = np.concatenate([grid[npad:0:-1], grid, grid[n_bins : n_bins - npad : -1]])
                density = ss.convolve(grid, kernel, mode="same", method="direct")[
                    npad : npad + n_bins
                ]
                norm_factor = len_x * d_x * (2 * np.pi * log_len_x ** 2 * scotts_factor ** 2) ** 0.5

                density /= norm_factor

                if cumulative:
                    density = density.cumsum() / density.sum()

                return density, xmin, xmax

            def _histogram(x, n_bins, range_hist=None):
                grid, _ = np.histogram(x, bins=n_bins, range=range_hist)
                return grid

            return _fast_kde(x)


class Fast_KDE_2d:
    def time_fast_kde_2d(self):
        try:
            x = np.random.randn(10000, 100)
            y = np.random.randn(10000, 100)
            import numba

            def _cov_1d(x):
                x = x - x.mean(axis=0)
                fact = x.shape[0] - 1
                by_hand = np.dot(x.T, x.conj()) / fact
                return np.array(by_hand)

            def _cov(data):
                if data.ndim == 1:
                    return _cov_1d(data)
                elif data.ndim == 2:
                    x = data.astype(float)
                    avg, _ = np.average(x, axis=1, weights=None, returned=True)
                    fact = x.shape[1] - 1
                    if fact <= 0:
                        warnings.warn(
                            "Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2
                        )
                        fact = 0.0
                    x -= avg[:, None]
                    x_t = x.T
                    c_c = _dot(x, x_t.conj())
                    c_c *= np.true_divide(1, fact)
                    return c_c.squeeze()
                else:
                    raise ValueError("{} dimension arrays are not supported".format(data.ndimn))

            @numba.njit(cache=True)
            def _dot(x, y):
                return np.dot(x, y)

            def _stack(x, y):
                return np.vstack((x, y))

            def _fast_kde_2d(x, y, gridsize=(128, 128), circular=False):
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
                grid = ss.convolve2d(grid, kernel, mode="same", boundary=boundary)

                norm_factor = np.linalg.det(2 * np.pi * cov * scotts_factor ** 2)
                norm_factor = len_x * d_x * d_y * norm_factor ** 0.5

                grid /= norm_factor

                return grid, xmin, xmax, ymin, ymax

            return _fast_kde_2d(x, y)

        except ImportError:
            x = np.random.randn(10000, 100)
            y = np.random.randn(10000, 100)

            def _cov(data):
                return np.cov(data)

            def _stack(x, y):
                return np.vstack((x, y))

            def _fast_kde_2d(x, y, gridsize=(128, 128), circular=False):
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
                kernel = np.dot(inv_cov, kernel) * kernel
                kernel = np.exp(-kernel.sum(axis=0) / 2)
                kernel = kernel.reshape((int(kern_ny), int(kern_nx)))

                boundary = "wrap" if circular else "symm"

                grid = coo_matrix((weights, xyi), shape=(n_x, n_y)).toarray()
                grid = ss.convolve2d(grid, kernel, mode="same", boundary=boundary)

                norm_factor = np.linalg.det(2 * np.pi * cov * scotts_factor ** 2)
                norm_factor = len_x * d_x * d_y * norm_factor ** 0.5

                grid /= norm_factor

                return grid, xmin, xmax, ymin, ymax

            return _fast_kde_2d(x, y)


class Data:
    def time_2d_custom(self):
        data = np.random.randn(100000)

        def two_de(data):
            if not isinstance(data, np.ndarray):
                return np.atleast_2d(data)
            if data.ndim == 0:
                result = data.reshape(1, 1)
            elif data.ndim == 1:
                result = data[newaxis, :]
            else:
                result = data
            return result

        return two_de(data)

    def time_numpy_2d(self):
        data = np.random.randn(100000)
        return np.atleast_2d(data)

    def time_1d_custom(self):
        x = np.random.randn(100000).tolist()

        def one_de(x):
            """Jitting numpy atleast_1d."""
            if not isinstance(x, np.ndarray):
                return np.atleast_1d(x)
            if x.ndim == 0:
                result = x.reshape(1)
            else:
                result = x
            return result

        return one_de(x)

    def time_numpy_1d(self):
        x = np.random.randn(100000).tolist()
        return np.atleast_1d(x)
