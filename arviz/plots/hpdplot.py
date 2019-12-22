"""Plot hpd intervals for regression data."""
import numpy as np
from scipy.interpolate import griddata
from scipy.signal import savgol_filter

from ..stats import hpd
from .plot_utils import get_plotting_function


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
    backend=None,
    backend_kwargs=None,
    show=None,
):
    r"""
    Plot hpd intervals for regression data.

    Parameters
    ----------
    x : array-like
        Values to plot
    y : array-like
        values from which to compute the hpd. Assumed shape (chain, draw, \*shape).
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
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : matplotlib axes or bokeh figures
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs.setdefault("color", color)
    plot_kwargs.setdefault("alpha", 0)

    if fill_kwargs is None:
        fill_kwargs = {}
    fill_kwargs.setdefault("color", color)
    fill_kwargs.setdefault("alpha", 0.5)

    x = np.asarray(x)
    y = np.asarray(y)

    x_shape = x.shape
    y_shape = y.shape
    if y_shape[-len(x_shape) :] != x_shape:
        msg = "Dimension mismatch for x: {} and y: {}."
        msg += " y-dimensions should be (chain, draw, *x.shape) or"
        msg += " (draw, *x.shape)"
        raise TypeError(msg.format(x_shape, y_shape))

    if len(y_shape[: -len(x_shape)]) > 1:
        new_shape = tuple([-1] + list(x_shape))
        y = y.reshape(new_shape)

    hpd_ = hpd(y, credible_interval=credible_interval, circular=circular, multimodal=False)

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

    hpdplot_kwargs = dict(
        ax=ax,
        x_data=x_data,
        y_data=y_data,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_hpd", "hpdplot", backend)
    ax = plot(**hpdplot_kwargs)
    return ax
