# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
import scipy.stats as st


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

            @numba.njit(cache=True)
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
            return st.circstd(data)
