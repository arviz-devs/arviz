# Write the benchmarking functions here.
# See "Writing benchmarks" in the airspeed velocity docs for more information.
# https://asv.readthedocs.io/en/stable/
import numpy as np
from scipy.stats import circstd
import arviz as az
from arviz.stats.stats_utils import _circular_standard_deviation, stats_variance_2d
from arviz.numeric_utils import _fast_kde, _fast_kde_2d, histogram


class Hist:
    params = (True, False)
    param_names = ("Numba",)

    def setup(self, numba_flag):
        self.data = np.random.rand(10000, 1000)
        if numba_flag:
            az.Numba.enable_numba()
        else:
            az.Numba.disable_numba()

    def time_histogram(self, numba_flag):
        histogram(self.data, bins=100)


class Variance:
    params = (True, False)
    param_names = ("Numba",)

    def setup(self, numba_flag):
        self.data = np.random.rand(10000, 1000)
        if numba_flag:
            az.Numba.enable_numba()
        else:
            az.Numba.disable_numba()

    def time_variance_2d(self, numba_flag):
        stats_variance_2d(self.data)


class CircStd:
    params = (True, False)
    param_names = ("Numba",)

    def setup(self, numba_flag):
        self.data = np.random.randn(10000, 1000)
        if numba_flag:
            self.circstd = _circular_standard_deviation
        else:
            self.circstd = circstd

    def time_circ_std(self, numba_flag):
        self.circstd(self.data)


class Fast_Kde_1d:
    params = [(True, False), (10 ** 5, 10 ** 6, 10 ** 7)]
    param_names = ("Numba", "n")

    def setup(self, numba_flag, n):
        self.x = np.random.randn(n // 10, 10)
        if numba_flag:
            az.Numba.enable_numba()
        else:
            az.Numba.disable_numba()

    def time_fast_kde_normal(self, numba_flag, n):
        _fast_kde(self.x)


class Fast_KDE_2d:
    params = [(True, False), ((100, 10 ** 4), (10 ** 4, 100), (1000, 1000))]
    param_names = ("Numba", "shape")

    def setup(self, numba_flag, shape):
        self.x = np.random.randn(*shape)
        self.y = np.random.randn(*shape)
        if numba_flag:
            az.Numba.enable_numba()
        else:
            az.Numba.disable_numba()

    def time_fast_kde_2d(self, numba_flag, shape):
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
