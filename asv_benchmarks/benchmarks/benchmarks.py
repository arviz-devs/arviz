# Write the benchmarking functions here.
# See "Writing benchmarks" in the airspeed velocity docs for more information.
# https://asv.readthedocs.io/en/stable/
import numpy as np
from numpy import newaxis
from scipy.stats import circstd
from scipy.sparse import coo_matrix
import scipy.signal as ss
import warnings
import arviz as az
from arviz.stats.stats_utils import _circular_standard_deviation
from arviz.plots.plot_utils import _fast_kde, _fast_kde_2d


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
    params = (True, False)
    param_names = "Numba",

    def setup(self, numba_flag):
        self.data = np.random.randn(10000, 1000)
        if numba_flag:
            self.circstd = _circular_standard_deviation
        else:
            self.circstd = circstd

    def time_circ_std(self, numba_flag):
        self.circstd(self.data)


class Fast_Kde_1d:
    params = [(True, False), (10**5, 10**6, 10**7)]
    param_names = ("Numba", "n")

    def setup(self, numba_flag, n):
        self.x = np.random.randn(n//10, 10)

    def time_fast_kde_normal(self, numba_flag, n):
        if numba_flag:
            az.Numba.enable_numba()
        else:
            az.Numba.disable_numba()

        _fast_kde(self.x)

class Fast_KDE_2d:
    params = [(True, False), (10**5, 10**6)]
    param_names = ("Numba", "n")

    def setup(self, numba_flag, n):
        self.x = np.random.randn(n//10, 10)
        self.y = np.random.randn(n//10, 10)

    def time_fast_kde_2d(self, numba_flag, n):
        if numba_flag:
            az.Numba.enable_numba()
        else:
            az.Numba.disable_numba()

        _fast_kde_2d(self.x, self.y)

class Atleast_Nd:
    params = ("az.utils", "numpy")
    param_names = ("source",)

    def setup(self, source):
        self.data = np.random.randn(100000)
        self.x = np.random.randn(100000).tolist()
        if source == "az.utils":
            self.atleast_2d = az.utils.two_de
            self.atleast_1d = az.utils.one_de
        else:
            self.atleast_2d = np.atleast_2d
            self.atleast_1d = np.atleast_1d

    def time_atleast_2d_array(self, source):
        self.atleast_2d(self.data)

    def time_atleast_1d(self, source):
        self.atleast_1d(self.x)
