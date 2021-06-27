"""Plot highest density intervals for regression data."""
import warnings

import numpy as np
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from xarray import Dataset

from ..rcparams import rcParams
from ..stats import hdi
from .plot_utils import get_plotting_function


def plot_hdi(
    x,
    y=None,
    hdi_prob=None,
    hdi_data=None,
    color="C1",
    circular=False,
    smooth=True,
    smooth_kwargs=None,
    figsize=None,
    fill_kwargs=None,
    plot_kwargs=None,
    hdi_kwargs=None,
    ax=None,
    backend=None,
    backend_kwargs=None,
    show=None,
):
    r"""
    Plot HDI intervals for regression data.

    Parameters
    ----------
    x : array-like
        Values to plot.
    y : array-like, optional
        Values from which to compute the HDI. Assumed shape ``(chain, draw, \*shape)``.
        Only optional if hdi_data is present.
    hdi_data : array_like, optional
        Precomputed HDI values to use. Assumed shape is ``(*x.shape, 2)``.
    hdi_prob : float, optional
        Probability for the highest density interval. Defaults to ``stats.hdi_prob`` rcParam.
    color : str, optional
        Color used for the limits of the HDI and fill. Should be a valid matplotlib color.
    circular : bool, optional
        Whether to compute the HDI taking into account `x` is a circular variable
        (in the range [-np.pi, np.pi]) or not. Defaults to False (i.e non-circular variables).
    smooth : boolean, optional
        If True the result will be smoothed by first computing a linear interpolation of the data
        over a regular grid and then applying the Savitzky-Golay filter to the interpolated data.
        Defaults to True.
    smooth_kwargs : dict, optional
        Additional keywords modifying the Savitzky-Golay filter. See
        :func:`scipy:scipy.signal.savgol_filter` for details.
    figsize : tuple
        Figure size. If None it will be defined automatically.
    fill_kwargs : dict, optional
        Keywords passed to :meth:`mpl:matplotlib.axes.Axes.fill_between`
        (use fill_kwargs={'alpha': 0} to disable fill) or to
        :meth:`bokeh:bokeh.plotting.figure.Figure.patch`.
    plot_kwargs : dict, optional
        HDI limits keyword arguments, passed to :meth:`mpl:matplotlib.axes.Axes.plot` or
        :meth:`bokeh:bokeh.plotting.figure.Figure.patch`.
    hdi_kwargs : dict, optional
        Keyword arguments passed to :func:`~arviz.hdi`. Ignored if ``hdi_data`` is present.
    ax : axes, optional
        Matplotlib axes or bokeh figures.
    backend : {"matplotlib","bokeh"}, optional
        Select plotting backend.
    backend_kwargs : bool, optional
        These are kwargs specific to the backend being used, passed to
        :meth:`mpl:matplotlib.axes.Axes.plot` or
        :meth:`bokeh:bokeh.plotting.figure.Figure.patch`.
    show : bool, optional
        Call backend show function.

    Returns
    -------
    axes : matplotlib axes or bokeh figures

    See Also
    --------
    hdi : Calculate highest density interval (HDI) of array for given probability.

    Examples
    --------
    Plot HDI interval of simulated regression data using `y` argument:

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import arviz as az
        >>> x_data = np.random.normal(0, 1, 100)
        >>> y_data = np.random.normal(2 + x_data * 0.5, 0.5, (2, 50, 100))
        >>> az.plot_hdi(x_data, y_data)

    ``plot_hdi`` can also be given precalculated values with the argument ``hdi_data``. This example
    shows how to use :func:`~arviz.hdi` to precalculate the values and pass these values to
    ``plot_hdi``. Similarly to an example in ``hdi`` we are using the ``input_core_dims``
    argument of :func:`~arviz.wrap_xarray_ufunc` to manually define the dimensions over which
    to calculate the HDI.

    .. plot::
        :context: close-figs

        >>> hdi_data = az.hdi(y_data, input_core_dims=[["draw"]])
        >>> ax = az.plot_hdi(x_data, hdi_data=hdi_data[0], color="r", fill_kwargs={"alpha": .2})
        >>> az.plot_hdi(x_data, hdi_data=hdi_data[1], color="k", ax=ax, fill_kwargs={"alpha": .2})

    ``plot_hdi`` can also be used with Inference Data objects. Here we use the posterior predictive
    to plot the HDI interval.

    .. plot::
        :context: close-figs

        >>> X = np.random.normal(0,1,100)
        >>> Y = np.random.normal(2 + X * 0.5, 0.5, (10,100))
        >>> idata = az.from_dict(posterior={"y": Y}, constant_data={"x":X})
        >>> x_data = idata.constant_data.x
        >>> y_data = idata.posterior.y
        >>> az.plot_hdi(x_data, y_data)

    """
    if hdi_kwargs is None:
        hdi_kwargs = {}

    x = np.asarray(x)
    x_shape = x.shape

    if y is None and hdi_data is None:
        raise ValueError("One of {y, hdi_data} is required")
    if hdi_data is not None and y is not None:
        warnings.warn("Both y and hdi_data arguments present, ignoring y")
    elif hdi_data is not None:
        hdi_prob = (
            hdi_data.hdi.attrs.get("hdi_prob", np.nan) if hasattr(hdi_data, "hdi") else np.nan
        )
        if isinstance(hdi_data, Dataset):
            data_vars = list(hdi_data.data_vars)
            if len(data_vars) != 1:
                raise ValueError(
                    "Found several variables in hdi_data. Only single variable Datasets are "
                    "supported."
                )
            hdi_data = hdi_data[data_vars[0]]
    else:
        y = np.asarray(y)
        if hdi_prob is None:
            hdi_prob = rcParams["stats.hdi_prob"]
        else:
            if not 1 >= hdi_prob > 0:
                raise ValueError("The value of hdi_prob should be in the interval (0, 1]")
        hdi_data = hdi(y, hdi_prob=hdi_prob, circular=circular, multimodal=False, **hdi_kwargs)

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
        color=color,
        figsize=figsize,
        plot_kwargs=plot_kwargs,
        fill_kwargs=fill_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_hdi", "hdiplot", backend)
    ax = plot(**hdiplot_kwargs)
    return ax
