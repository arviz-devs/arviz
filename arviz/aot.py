import numpy as np
import numba
from numba.pycc import CC

module = CC("numba_compilations")


@module.export("stats_variance_1d", "(f8[:],i8)")
def stats_variance_1d(data, ddof=0):
    """Pre compiled method to get 1d variance."""
    a_a, b_b = 0, 0
    for i in data:
        a_a = a_a + i
        b_b = b_b + i * i
    var = b_b / (len(data)) - ((a_a / (len(data))) ** 2)
    var = var * (len(data) / (len(data) - ddof))
    return var


@numba.njit
def _stats_variance_1d_2d(data, ddof=0):
    a_a, b_b = 0, 0
    for i in data:
        a_a = a_a + i
        b_b = b_b + i * i
    var = b_b / (len(data)) - ((a_a / (len(data))) ** 2)
    var = var * (len(data) / (len(data) - ddof))
    return var


@module.export("stats_variance_2d", "f8[:](f8[:,:],i8,i8)")
def stats_variance_2d(data, ddof=0, axis=1):
    """Pre compiled method to get 2d variance."""
    a_a, b_b = data.shape
    if axis == 1:
        var = np.zeros(a_a)
        for i in range(a_a):
            var[i] = _stats_variance_1d_2d(data[i], ddof=ddof)
        return var
    else:
        var = np.zeros(b_b)
        for i in range(b_b):
            var[i] = _stats_variance_1d_2d(data[:, i], ddof=ddof)
        return var


if __name__ == "__main__":
    module.compile()
