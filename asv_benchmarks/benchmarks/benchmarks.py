# Write the benchmarking functions here.
# See "Writing benchmarks" in the airspeed velocity docs for more information.
# https://asv.readthedocs.io/en/stable/
import numpy as np
from numpy import newaxis
from scipy.stats import circstd
from scipy.sparse import coo_matrix
import scipy.signal as ss
import warnings
from arviz.plots.plot_utils import _fast_kde, _fast_kde_2d
from arviz import Numba


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
    params = [(True, False), (10**3, 10**5, 10**6)]
    param_names = ("Numba", "n")

    def setup(self, numba_flag, n):
        self.x = np.random.randn(n, 10)

    def time_fast_kde_normal(self, numba_flag, n):
        if numba_flag:
            Numba.enable_numba()
        else:
            Numba.disable_numba()

        _fast_kde(self.x)

class Fast_KDE_2d:
    params = [(True, False), (10**3, 10**5)]
    param_names = ("Numba", "n")

    def setup(self, numba_flag, n):
        self.x = np.random.randn(n, 10)
        self.y = np.random.randn(n, 10)

    def time_fast_kde_2d(self, numba_flag, n):
        if numba_flag:
            Numba.enable_numba()
        else:
            Numba.disable_numba()

        _fast_kde_2d(self.x, self.y)

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
