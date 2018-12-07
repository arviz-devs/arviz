"""Plot hpd intervals for regression data."""
import numpy as np
from matplotlib.pyplot import gca
from scipy.interpolate import griddata
from scipy.signal import savgol_filter

from ..stats import hpd


def plot_hpd(
    x,
    y,
    credible_interval=0.94,
    color="C1",
    circular=False,
    smooth=True,
    smooth_kwargs=None,
    fill_kwargs=None,
    plot_kwargs=None,
    ax=None,
):
    """
    Plot hpd intervals for regression data.

    Parameters
    ----------
    x : array-like
        Values to plot
    y : array-like
        values ​​from which to compute the hpd
    credible_interval : float, optional
        Credible interval to plot. Defaults to 0.94.
    color : str
        Color used for the limits of the HPD interval and fill. Should be a valid matplotlib color
    circular : bool, optional
        Whether to compute the hpd taking into account `x` is a circular variable
        (in the range [-np.pi, np.pi]) or not. Defaults to False (i.e non-circular variables).
    smooth : boolean
        If True the result will be smoothed by first computing a linear interpolation of the data
        over a regular grid and then applying the Savitzky-Golay filter to the interpolated data.
        Defaults to True.
    smooth_kwargs : dict, optional
        Additional keywords modifying the Savitzky-Golay filter. See Scipy's documentation for
        details
    fill_kwargs : dict
        Keywords passed to `fill_between` (use fill_kwargs={'alpha': 0} to disable fill).
    plot_kwargs : dict
        Keywords passed to HPD limits
    ax : matplotlib axes

    Returns
    -------
    ax : matplotlib axes
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs.setdefault("color", color)
    plot_kwargs.setdefault("alpha", 0)

    if fill_kwargs is None:
        fill_kwargs = {}
    fill_kwargs.setdefault("color", color)
    fill_kwargs.setdefault("alpha", 0.5)

    if ax is None:
        ax = gca()

    hpd_ = hpd(y, credible_interval=credible_interval, circular=circular)

    if smooth:
        if smooth_kwargs is None:
            smooth_kwargs = {}
        smooth_kwargs.setdefault("window_length", 55)
        smooth_kwargs.setdefault("polyorder", 2)
        x_data = np.linspace(x.min(), x.max(), 200)
        hpd_interp = griddata(x, hpd_, x_data)
        y_data = savgol_filter(hpd_interp, axis=0, **smooth_kwargs)
    else:
        idx = np.argsort(x)
        x_data = x[idx]
        y_data = hpd_[idx]

    ax.plot(x_data, y_data, **plot_kwargs)
    ax.fill_between(x_data, y_data[:, 0], y_data[:, 1], **fill_kwargs)

    return ax
