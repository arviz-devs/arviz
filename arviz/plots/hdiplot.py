"""Plot highest density intervals for regression data."""
import warnings

import numpy as np
from scipy.interpolate import griddata
from scipy.signal import savgol_filter

from ..stats import hdi
from .plot_utils import get_plotting_function, matplotlib_kwarg_dealiaser
from ..rcparams import rcParams
from ..utils import credible_interval_warning


def plot_hdi(
    x,
    y,
    hdi_prob=None,
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
    credible_interval=None,
):
    r"""
    Plot hdi intervals for regression data.

    Parameters
    ----------
    x : array-like
        Values to plot
    y : array-like
        values from which to compute the hdi. Assumed shape (chain, draw, \*shape).
    hdi_prob : float, optional
        Probability for the highest density interval. Defaults to 0.94.
    color : str
        Color used for the limits of the hdi and fill. Should be a valid matplotlib color
    circular : bool, optional
        Whether to compute the hdi taking into account `x` is a circular variable
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
        Keywords passed to hdi limits
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show : bool, optional
        Call backend show function.
    credible_interval: float, optional
        deprecated: Please see hdi_prob

    Returns
    -------
    axes : matplotlib axes or bokeh figures
    """
    if credible_interval:
        hdi_prob = credible_interval_warning(credible_interval, hdi_prob)

    plot_kwargs = matplotlib_kwarg_dealiaser(plot_kwargs, "plot")
    plot_kwargs.setdefault("color", color)
    plot_kwargs.setdefault("alpha", 0)

    fill_kwargs = matplotlib_kwarg_dealiaser(fill_kwargs, "hexbin")
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

    if hdi_prob is None:
        hdi_prob = rcParams["stats.hdi_prob"]
    else:
        if not 1 >= hdi_prob > 0:
            raise ValueError("The value of hdi_prob should be in the interval (0, 1]")

    hdi_ = hdi(y, hdi_prob=hdi_prob, circular=circular, multimodal=False)

    if smooth:
        if smooth_kwargs is None:
            smooth_kwargs = {}
        smooth_kwargs.setdefault("window_length", 55)
        smooth_kwargs.setdefault("polyorder", 2)
        x_data = np.linspace(x.min(), x.max(), 200)
        x_data[0] = (x_data[0] + x_data[1]) / 2
        hdi_interp = griddata(x, hdi_, x_data)
        y_data = savgol_filter(hdi_interp, axis=0, **smooth_kwargs)
    else:
        idx = np.argsort(x)
        x_data = x[idx]
        y_data = hdi_[idx]

    hdiplot_kwargs = dict(
        ax=ax,
        x_data=x_data,
        y_data=y_data,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_hdi", "hdiplot", backend)
    ax = plot(**hdiplot_kwargs)
    return ax


def plot_hpd(*args, **kwargs):  # noqa: D103
    warnings.warn("plot_hdi has been deprecated, please use plot_hdi", DeprecationWarning)
    return plot_hdi(*args, **kwargs)
