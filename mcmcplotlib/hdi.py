from __future__ import division
import numpy as np
import scipy.stats.kde as kde

def hdi_grid(trace, cred_mass=0.95, roundto=3):
    """Computes Highest Density Interval (HDI)"""
    density = kde.gaussian_kde(trace)
    # get upper and lower bounds
    l = np.min(trace)
    u = np.max(trace)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    mode = x[np.argmax(y)]
    diff = (u-l)/20  # differences of 5%
    normalization_factor = np.sum(y)
    xy = zip(x, y/normalization_factor)
    xy.sort(key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= cred_mass:
            break
    hdv.sort()
    hdi = []
    hdi.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)-1):
        if hdv[i]-hdv[i-1] >= diff:
            hdi.append(round(hdv[i-1], roundto))
            hdi.append(round(hdv[i], roundto))
    hdi.append(round(max(hdv), roundto))
    ite = iter(hdi)
    hdi = zip(ite, ite)
    modes = []
    for value in hdi:
        l = np.min(value[0])
        u = np.max(value[1])
        xi = np.linspace(l, u, 100)
        yi = density.evaluate(xi)
        modes.append(round(xi[np.argmax(yi)], roundto))
    return hdi, x, y, modes
