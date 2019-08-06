from numba.pycc import CC
import numba
import numpy as np

module = CC('numba-compilations')


@numba.njit
def stats_variance_1d(data, ddof=0):
    a_a, b_b = 0, 0
    for i in data:
        a_a = a_a + i
        b_b = b_b + i * i
    var = b_b / (len(data)) - ((a_a / (len(data))) ** 2)
    var = var * (len(data) / (len(data) - ddof))
    return var


@module.export('stats_variance_2d', 'f8[:](f8[:],i8,i8)')
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


# Remember to flatten the array before when using this method
@module.export('histogram_stats', 'f8[:](f8[:],f8[:]')
def histogram_stats(data, bins):
    return np.histogram(data, bins=bins)[0]


# Remember to flatten the array before when using this method
@module.export('histogram_kde', 'f8[:](f8[:],i8,(f8,f8)')
def histogram_plots(x, n_bins, range_hist=None):
    grid, _ = np.histogram(x, bins=n_bins, range=range_hist)
    return grid


@module.export('dot', 'f8[:], f8[:]')
@module.export('dot', 'f8[:,:], f8[:,:]')
def dot(x, y):
    return np.dot(x, y)

