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
    y=None,
    hdi_prob=None,
    hdi_data=None,
    color="C1",
    circular=False,
    smooth=True,
    smooth_kwargs=None,
    fill_kwargs=None,
    plot_kwargs=None,
    hdi_kwargs=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
    credible_interval=None,
):
    r"""
    Plot HDI intervals for regression data.

    Parameters
    ----------
    x : array-like
        Values to plot.
    y : array-like, optional
        Values from which to compute the HDI. Assumed shape (chain, draw, \*shape).
        Only optional if hdi_data is present.
    hdi_data : array_like, optional
        HDI values to use.
    hdi_prob : float, optional
        Probability for the highest density interval. Defaults to 0.94.
    color : str
        Color used for the limits of the HDI and fill. Should be a valid matplotlib color.
    circular : bool, optional
        Whether to compute the HDI taking into account `x` is a circular variable
        (in the range [-np.pi, np.pi]) or not. Defaults to False (i.e non-circular variables).
    smooth : boolean
        If True the result will be smoothed by first computing a linear interpolation of the data
        over a regular grid and then applying the Savitzky-Golay filter to the interpolated data.
        Defaults to True.
    smooth_kwargs : dict, optional
        Additional keywords modifying the Savitzky-Golay filter. See Scipy's documentation for
        details.
    fill_kwargs : dict
        Keywords passed to `fill_between` (use fill_kwargs={'alpha': 0} to disable fill).
    plot_kwargs : dict
        Keywords passed to HDI limits.
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
        Deprecated: Please see hdi_prob

    Returns
    -------
    axes : matplotlib axes or bokeh figures
    """
    if credible_interval:
        hdi_prob = credible_interval_warning(credible_interval, hdi_prob)

    if hdi_kwargs is None:
        hdi_kwargs = {}
    plot_kwargs = matplotlib_kwarg_dealiaser(plot_kwargs, "plot")
    plot_kwargs.setdefault("color", color)
    plot_kwargs.setdefault("alpha", 0)

    fill_kwargs = matplotlib_kwarg_dealiaser(fill_kwargs, "hexbin")
    fill_kwargs.setdefault("color", color)
    fill_kwargs.setdefault("alpha", 0.5)

    x = np.asarray(x)
    x_shape = x.shape


    if y is None and hdi_data is None:
        raise ValueError("One of {y, hdi_data} is required")
    elif hdi_data is not None and y is not None:
        warnings.warn("Both y and hdi_data arguments present, ignoring y")
    elif y is not None:
        y = np.asarray(y)
        if hdi_prob is None:
            hdi_prob = rcParams["stats.hdi_prob"]
        else:
            if not 1 >= hdi_prob > 0:
                raise ValueError("The value of hdi_prob should be in the interval (0, 1]")
        hdi_data = hdi(y, hdi_prob=hdi_prob, circular=circular, multimodal=False, **hdi_kwargs)
    else:
        hdi_prob = hdi_data.hdi.attrs.get("hdi_prob", np.nan)

    hdi_shape = hdi_data.shape
    if hdi_shape[:-1] != x_shape:
        msg = (
            "Dimension mismatch for x: {} and hdi: {}. Check the dimensions of y and"
            "hdi_kwargs to make sure they are compatible"
        )
        raise TypeError(msg.format(x_shape, hdi_shape))

    if smooth:
        if smooth_kwargs is None:
            smooth_kwargs = {}
        smooth_kwargs.setdefault("window_length", 55)
        smooth_kwargs.setdefault("polyorder", 2)
        x_data = np.linspace(x.min(), x.max(), 200)
        x_data[0] = (x_data[0] + x_data[1]) / 2
        hdi_interp = griddata(x, hdi_data, x_data)
        y_data = savgol_filter(hdi_interp, axis=0, **smooth_kwargs)
    else:
        idx = np.argsort(x)
        x_data = x[idx]
        y_data = hdi_data[idx]

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
    warnings.warn("plot_hpd has been deprecated, please use plot_hdi", DeprecationWarning)
    return plot_hdi(*args, **kwargs)
