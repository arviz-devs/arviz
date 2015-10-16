from __future__ import division
import numpy as np
import scipy.stats.kde as kde

def hdi_grid(trace, cred_mass=0.95, roundto=2):
    """Computes Highest Density Interval (HDI)"""
    density = kde.gaussian_kde(trace)
    # get upper and lower bounds
    l = np.min(trace)
    u = np.max(trace)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= cred_mass:
            break
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hdi = []
    hdi.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hdi.append(round(hdv[i-1], roundto))
            hdi.append(round(hdv[i], roundto))
    hdi.append(round(max(hdv), roundto))
    ite = iter(hdi)
    hdi = list(zip(ite, ite))
    modes = []
    for value in hdi:
         x_hdi = x[(x > value[0]) & (x < value[1])]
         y_hdi = y[(x > value[0]) & (x < value[1])]
         modes.append(round(x_hdi[np.argmax(y_hdi)], roundto))
    return hdi, x, y, modes
