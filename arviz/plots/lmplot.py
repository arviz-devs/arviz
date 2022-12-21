"""Plot regression figure."""
import warnings
from numbers import Integral
from itertools import repeat

import xarray as xr
import numpy as np
from xarray.core.dataarray import DataArray

from ..sel_utils import xarray_var_iter
from ..rcparams import rcParams
from .plot_utils import default_grid, filter_plotters_list, get_plotting_function


def _repeat_flatten_list(lst, n):
    return [item for sublist in repeat(lst, n) for item in sublist]


def plot_lm(
    y,
    idata=None,
    x=None,
    y_model=None,
    y_hat=None,
    num_samples=50,
    kind_pp="samples",
    kind_model="lines",
    xjitter=False,
    plot_dim=None,
    backend=None,
    y_kwargs=None,
    y_hat_plot_kwargs=None,
    y_hat_fill_kwargs=None,
    y_model_plot_kwargs=None,
    y_model_fill_kwargs=None,
    y_model_mean_kwargs=None,
    backend_kwargs=None,
    show=None,
    figsize=None,
    textsize=None,
    axes=None,
    legend=True,
    grid=True,
):
    """Posterior predictive and mean plots for regression-like data.

    Parameters
    ----------
    y : str or DataArray or ndarray
        If str, variable name from ``observed_data``.
    idata : InferenceData, Optional
        Optional only if ``y`` is not str.
    x : str, tuple of strings, DataArray or array-like, optional
        If str or tuple, variable name from ``constant_data``.
        If ndarray, could be 1D, or 2D for multiple plots.
        If None, coords name of ``y`` (``y`` should be DataArray).
    y_model : str or Sequence, Optional
        If str, variable name from ``posterior``.
        Its dimensions should be same as ``y`` plus added chains and draws.
    y_hat : str, Optional
        If str, variable name from ``posterior_predictive``.
        Its dimensions should be same as ``y`` plus added chains and draws.
    num_samples : int, Optional, Default 50
        Significant if ``kind_pp`` is "samples" or ``kind_model`` is "lines".
        Number of samples to be drawn from posterior predictive or
    kind_pp : {"samples", "hdi"}, Default "samples"
        Options to visualize uncertainty in data.
    kind_model : {"lines", "hdi"}, Default "lines"
        Options to visualize uncertainty in mean of the data.
    plot_dim : str, Optional
        Necessary if ``y`` is multidimensional.
    backend : str, Optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    y_kwargs : dict, optional
        Passed to :meth:`mpl:matplotlib.axes.Axes.plot` in matplotlib
        and :meth:`bokeh:bokeh.plotting.Figure.circle` in bokeh
    y_hat_plot_kwargs : dict, optional
        Passed to :meth:`mpl:matplotlib.axes.Axes.plot` in matplotlib
        and :meth:`bokeh:bokeh.plotting.Figure.circle` in bokeh
    y_hat_fill_kwargs : dict, optional
        Passed to :func:`arviz.plot_hdi`
    y_model_plot_kwargs : dict, optional
        Passed to :meth:`mpl:matplotlib.axes.Axes.plot` in matplotlib
        and :meth:`bokeh:bokeh.plotting.Figure.line` in bokeh
    y_model_fill_kwargs : dict, optional
        Significant if ``kind_model`` is "hdi". Passed to :func:`arviz.plot_hdi`
    y_model_mean_kwargs : dict, optional
        Passed to :meth:`mpl:matplotlib.axes.Axes.plot` in matplotlib
        and :meth:`bokeh:bokeh.plotting.Figure.line` in bokeh
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used. Passed to
        :func:`matplotlib.pyplot.subplots` or
        :func:`bokeh.plotting.figure`.
    figsize : (float, float), optional
        Figure size. If None it will be defined automatically.
    textsize : float, optional
        Text size scaling factor for labels, titles and lines. If None it will be
        autoscaled based on ``figsize``.
    axes : 2D numpy array-like of matplotlib_axes or bokeh_figures, optional
        A 2D array of locations into which to plot the densities. If not supplied, Arviz will create
        its own array of plot areas (and return it).
    show : bool, optional
        Call backend show function.
    legend : bool, optional
        Add legend to figure. By default True.
    grid : bool, optional
        Add grid to figure. By default True.


    Returns
    -------
    axes: matplotlib axes or bokeh figures

    See Also
    --------
    plot_ts : Plot timeseries data
    plot_ppc : Plot for posterior/prior predictive checks

    Examples
    --------
    Plot regression default plot

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> import numpy as np
        >>> import xarray as xr
        >>> idata = az.load_arviz_data('regression1d')
        >>> x = xr.DataArray(np.linspace(0, 1, 100))
        >>> idata.posterior["y_model"] = idata.posterior["intercept"] + idata.posterior["slope"]*x
        >>> az.plot_lm(idata=idata, y="y", x=x)

    Plot regression data and mean uncertainty

    .. plot::
        :context: close-figs

        >>> az.plot_lm(idata=idata, y="y", x=x, y_model="y_model")

    Plot regression data and mean uncertainty in hdi form

    .. plot::
        :context: close-figs

        >>> az.plot_lm(
        ...     idata=idata, y="y", x=x, y_model="y_model", kind_pp="hdi", kind_model="hdi"
        ... )

    Plot regression data for multi-dimensional y using plot_dim

    .. plot::
        :context: close-figs

        >>> data = az.from_dict(
        ...     observed_data = { "y": np.random.normal(size=(5, 7)) },
        ...     posterior_predictive = {"y": np.random.randn(4, 1000, 5, 7) / 2},
        ...     dims={"y": ["dim1", "dim2"]},
        ...     coords={"dim1": range(5), "dim2": range(7)}
        ... )
        >>> az.plot_lm(idata=data, y="y", plot_dim="dim1")
    """
    if kind_pp not in ("samples", "hdi"):
        raise ValueError("kind_ppc should be either samples or hdi")

    if kind_model not in ("lines", "hdi"):
        raise ValueError("kind_model should be either lines or hdi")

    if y_hat is None and isinstance(y, str):
        y_hat = y

    if isinstance(y, str):
        y = idata.observed_data[y]
    elif not isinstance(y, DataArray):
        y = xr.DataArray(y)

    if len(y.dims) > 1 and plot_dim is None:
        raise ValueError("Argument plot_dim is needed in case of multidimensional data")

    x_var_names = None
    if isinstance(x, str):
        x = idata.constant_data[x]
        x_skip_dims = x.dims
    elif isinstance(x, tuple):
        x_var_names = x
        x = idata.constant_data
        x_skip_dims = x.dims
    elif isinstance(x, DataArray):
        x_skip_dims = x.dims
    elif x is None:
        x = y.coords[y.dims[0]] if plot_dim is None else y.coords[plot_dim]
        x_skip_dims = x.dims
    else:
        x = xr.DataArray(x)
        x_skip_dims = [x.dims[-1]]

    # If posterior is present in idata and y_hat is there, get its values
    if isinstance(y_model, str):
        if "posterior" not in idata.groups():
            warnings.warn("Posterior not found in idata", UserWarning)
            y_model = None
        elif hasattr(idata.posterior, y_model):
            y_model = idata.posterior[y_model]
        else:
            warnings.warn("y_model not found in posterior", UserWarning)
            y_model = None

    # If posterior_predictive is present in idata and y_hat is there, get its values
    if isinstance(y_hat, str):
        if "posterior_predictive" not in idata.groups():
            warnings.warn("posterior_predictive not found in idata", UserWarning)
            y_hat = None
        elif hasattr(idata.posterior_predictive, y_hat):
            y_hat = idata.posterior_predictive[y_hat]
        else:
            warnings.warn("y_hat not found in posterior_predictive", UserWarning)
            y_hat = None

    # Check if num_pp_smaples is valid and generate num_pp_smaples number of random indexes.
    # Only needed if kind_pp="samples" or kind_model="lines". Not req for plotting hdi
    pp_sample_ix = None
    if (y_hat is not None and kind_pp == "samples") or (
        y_model is not None and kind_model == "lines"
    ):
        if y_hat is not None:
            total_pp_samples = y_hat.sizes["chain"] * y_hat.sizes["draw"]
        else:
            total_pp_samples = y_model.sizes["chain"] * y_model.sizes["draw"]

        if (
            not isinstance(num_samples, Integral)
            or num_samples < 1
            or num_samples > total_pp_samples
        ):
            raise TypeError(f"`num_samples` must be an integer between 1 and {total_pp_samples}.")

        pp_sample_ix = np.random.choice(total_pp_samples, size=num_samples, replace=False)

    # crucial step in case of multidim y
    if plot_dim is None:
        skip_dims = list(y.dims)
    elif isinstance(plot_dim, str):
        skip_dims = [plot_dim]
    elif isinstance(plot_dim, tuple):
        skip_dims = list(plot_dim)

    # Generate x axis plotters.
    x = filter_plotters_list(
        plotters=list(
            xarray_var_iter(
                x,
                var_names=x_var_names,
                skip_dims=set(x_skip_dims),
                combined=True,
            )
        ),
        plot_kind="plot_lm",
    )

    # Generate y axis plotters
    y = filter_plotters_list(
        plotters=list(
            xarray_var_iter(
                y,
                skip_dims=set(skip_dims),
                combined=True,
            )
        ),
        plot_kind="plot_lm",
    )

    # If there are multiple x and multidimensional y, we need total of len(x)*len(y) graphs
    len_y = len(y)
    len_x = len(x)
    length_plotters = len_x * len_y
    y = _repeat_flatten_list(y, len_x)
    x = _repeat_flatten_list(x, len_y)

    # Filter out the required values to generate plotters
    if y_hat is not None:
        if kind_pp == "samples":
            y_hat = y_hat.stack(__sample__=("chain", "draw"))[..., pp_sample_ix]
            skip_dims += ["__sample__"]

        y_hat = [
            tup
            for _, tup in zip(
                range(len_y),
                xarray_var_iter(
                    y_hat,
                    skip_dims=set(skip_dims),
                    combined=True,
                ),
            )
        ]

        y_hat = _repeat_flatten_list(y_hat, len_x)

    # Filter out the required values to generate plotters
    if y_model is not None:
        if kind_model == "lines":
            y_model = y_model.stack(__sample__=("chain", "draw"))[..., pp_sample_ix]

        y_model = [
            tup
            for _, tup in zip(
                range(len_y),
                xarray_var_iter(
                    y_model,
                    skip_dims=set(y_model.dims),
                    combined=True,
                ),
            )
        ]
        y_model = _repeat_flatten_list(y_model, len_x)

    rows, cols = default_grid(length_plotters)

    lmplot_kwargs = dict(
        x=x,
        y=y,
        y_model=y_model,
        y_hat=y_hat,
        num_samples=num_samples,
        kind_pp=kind_pp,
        kind_model=kind_model,
        length_plotters=length_plotters,
        xjitter=xjitter,
        rows=rows,
        cols=cols,
        y_kwargs=y_kwargs,
        y_hat_plot_kwargs=y_hat_plot_kwargs,
        y_hat_fill_kwargs=y_hat_fill_kwargs,
        y_model_plot_kwargs=y_model_plot_kwargs,
        y_model_fill_kwargs=y_model_fill_kwargs,
        y_model_mean_kwargs=y_model_mean_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
        figsize=figsize,
        textsize=textsize,
        axes=axes,
        legend=legend,
        grid=grid,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_lm", "lmplot", backend)
    ax = plot(**lmplot_kwargs)
    return ax
