"""Plot timeseries data."""
import warnings
import numpy as np

from ..sel_utils import xarray_var_iter
from ..rcparams import rcParams
from .plot_utils import default_grid, get_plotting_function


def plot_ts(
    idata,
    y,
    x=None,
    y_hat=None,
    y_holdout=None,
    x_holdout=None,
    y_forecasts=None,
    plot_dim=None,
    holdout_dim=None,
    num_samples=100,
    backend=None,
    backend_kwargs=None,
    y_kwargs=None,
    y_hat_plot_kwargs=None,
    y_mean_plot_kwargs=None,
    vline_kwargs=None,
    textsize=None,
    figsize=None,
    legend=True,
    grid=False,
    axes=None,
    show=None,
):
    """Plot timeseries and it's components.

    Parameters
    ----------
    idata : InferenceData
    y : str, variable name from observed_data
    x : str, Optional
        If none, coords name of y dims
    y_hat : str, from posterior_predictive, optional
        Assumed to be of shape (chain, draw, *y.dims)
    y_holdout : str, from observed_data, optional
    x_holdout : str, from constant_data or y_holdout coords, optional
    y_forecasts : str, from posterior_predictive, optional
    plot_dim: str, Optional
        Necessary if y is multidimensional.
    holdout_dim: str, Optional
        Necessary if y_holdout or y_forecasts is multidimensional.
    num_samples : int, Optional, Default 100
    backend : str, Optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs : dict, optional
    figsize : tuple, optional
        Figure size. If None it will be defined automatically.
    textsize : float, optional
        Text size scaling factor for labels, titles and lines. If None it will be
        autoscaled based on figsize.


    Returns
    -------
    axes: matplotlib axes or bokeh figures.
    """
    # Assign default values if none is provided
    y_hat = y if y_hat is None and isinstance(y, str) else y_hat
    y_forecasts = y_holdout if y_forecasts is None and isinstance(y_holdout, str) else y_forecasts
    holdout_dim = plot_dim if holdout_dim is None and plot_dim is not None else holdout_dim

    if isinstance(y, str):
        y = idata.observed_data[y]

    if len(y.dims) > 1 and plot_dim is None:
        raise ValueError("Argument plot_dim is needed in case of multidimensional data")

    # Assigning values to x
    x_var_names = None
    if isinstance(x, str):
        x = idata.constant_data[x]
    elif isinstance(x, tuple):
        x_var_names = x
        x = idata.constant_data
    elif x is None:
        if plot_dim is None:
            x = y.coords[y.dims[0]]
        else:
            x = y.coords[plot_dim]
    else:
        TypeError("Invalid datatype for x")

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

    # If posterior_predictive is present in idata and y_forecasts is there, get its values
    x_holdout_var_names = None
    if isinstance(y_forecasts, str):
        if "posterior_predictive" not in idata.groups():
            warnings.warn("posterior_predictive not found in idata", UserWarning)
            y_forecasts = None
        elif hasattr(idata.posterior_predictive, y_forecasts):
            y_forecasts = idata.posterior_predictive[y_forecasts]
        else:
            warnings.warn("y_hat not found in posterior_predictive", UserWarning)
            y_forecasts = None

        # Assign values to x_holdout
        if x_holdout is None:
            if holdout_dim is None:
                x_holdout = y_forecasts.coords[y_forecasts.dims[-1]]
            else:
                x_holdout = y_forecasts.coords[holdout_dim]
        elif isinstance(x_holdout, str):
            x_holdout = idata.constant_data[x_holdout]
        elif isinstance(x_holdout, tuple):
            x_holdout_var_names = x_holdout
            x_holdout = idata.constant_data

    # Assign values to y_holdout
    if isinstance(y_holdout, str):
        y_holdout = idata.observed_data[y_holdout]
        if len(y_holdout.dims) > 1 and holdout_dim is None:
            raise ValueError("Argument holdout_dim is needed in case of multidimentional data")

        # Assign values to x_holdout
        if x_holdout is None:
            if holdout_dim is None:
                x_holdout = y_holdout.coords[y_holdout.dims[-1]]
            else:
                x_holdout = y_holdout.coords[holdout_dim]
        elif isinstance(x_holdout, str):
            x_holdout = idata.constant_data[x_holdout]
        elif isinstance(x_holdout, tuple):
            x_holdout_var_names = x_holdout
            x_holdout = idata.constant_data

    # Choose dims to generate y plotters
    if plot_dim is None:
        skip_dims = list(y.dims)
    elif isinstance(plot_dim, str):
        skip_dims = [plot_dim]
    elif isinstance(plot_dim, tuple):
        skip_dims = list(plot_dim)

    # Choose dims to generate y_holdout plotters
    if holdout_dim is None:
        if y_holdout is not None:
            skip_holdout_dims = list(y_holdout.dims)
        elif y_forecasts is not None:
            skip_holdout_dims = list(y_forecasts.dims)
    elif isinstance(holdout_dim, str):
        skip_holdout_dims = [holdout_dim]
    elif isinstance(holdout_dim, tuple):
        skip_holdout_dims = list(holdout_dim)

    y_plotters = list(
        xarray_var_iter(
            y,
            skip_dims=set(skip_dims),
            combined=True,
        )
    )

    x_plotters = list(
        xarray_var_iter(
            x,
            var_names=x_var_names,
            skip_dims=set(x.dims),
            combined=True,
        )
    )
    # Necessary when multidim y
    # If there are multiple x and multidimensional y, we need total of len(x)*len(y) graphs
    len_y = len(y_plotters)
    len_x = len(x_plotters)
    length_plotters = len_x * len_y
    y_plotters = np.tile(y_plotters, (len_x, 1))
    x_plotters = np.tile(x_plotters, (len_y, 1))

    # Generate plotters for all the available data
    y_mean_plotters = None
    if y_hat is not None:
        total_samples = y_hat.sizes["chain"] * y_hat.sizes["draw"]
        pp_sample_ix = np.random.choice(total_samples, size=num_samples, replace=False)

        y_hat_satcked = y_hat.stack(__sample__=("chain", "draw"))[..., pp_sample_ix]

        y_hat_plotters = list(
            xarray_var_iter(
                y_hat_satcked,
                skip_dims=set(skip_dims + ["__sample__"]),
                combined=True,
            )
        )

        y_mean = y_hat.mean(("chain", "draw"))
        y_mean_plotters = list(
            xarray_var_iter(
                y_mean,
                skip_dims=set(skip_dims),
                combined=True,
            )
        )

        # Necessary when multidim y
        # If there are multiple x and multidimensional y, we need total of len(x)*len(y) graphs
        y_hat_plotters = np.tile(y_hat_plotters, (len_x, 1))
        y_mean_plotters = np.tile(y_mean_plotters, (len_x, 1))

    y_holdout_plotters = None
    x_holdout_plotters = None
    if y_holdout is not None:
        y_holdout_plotters = list(
            xarray_var_iter(
                y_holdout,
                skip_dims=set(skip_holdout_dims),
                combined=True,
            )
        )

        x_holdout_plotters = list(
            xarray_var_iter(
                x_holdout,
                var_names=x_holdout_var_names,
                skip_dims=set(x_holdout.dims),
                combined=True,
            )
        )

        # Necessary when multidim y
        # If there are multiple x and multidimensional y, we need total of len(x)*len(y) graphs
        y_holdout_plotters = np.tile(y_holdout_plotters, (len_x, 1))
        x_holdout_plotters = np.tile(x_holdout_plotters, (len_y, 1))

    y_forecasts_plotters = None
    y_forecasts_mean_plotters = None
    if y_forecasts is not None:
        total_samples = y_forecasts.sizes["chain"] * y_forecasts.sizes["draw"]
        pp_sample_ix = np.random.choice(total_samples, size=num_samples, replace=False)

        y_forecasts_satcked = y_forecasts.stack(__sample__=("chain", "draw"))[..., pp_sample_ix]

        y_forecasts_plotters = list(
            xarray_var_iter(
                y_forecasts_satcked,
                skip_dims=set(skip_holdout_dims + ["__sample__"]),
                combined=True,
            )
        )

        y_forecasts_mean = y_forecasts.mean(("chain", "draw"))
        y_forecasts_mean_plotters = list(
            xarray_var_iter(
                y_forecasts_mean,
                skip_dims=set(skip_holdout_dims),
                combined=True,
            )
        )

        x_holdout_plotters = list(
            xarray_var_iter(
                x_holdout,
                var_names=x_holdout_var_names,
                skip_dims=set(x_holdout.dims),
                combined=True,
            )
        )

        # Necessary when multidim y
        # If there are multiple x and multidimensional y, we need total of len(x)*len(y) graphs
        y_forecasts_mean_plotters = np.tile(y_forecasts_mean_plotters, (len_x, 1))
        y_forecasts_plotters = np.tile(y_forecasts_plotters, (len_x, 1))
        x_holdout_plotters = np.tile(x_holdout_plotters, (len_y, 1))

    rows, cols = default_grid(length_plotters)

    tsplot_kwargs = dict(
        x_plotters=x_plotters,
        y_plotters=y_plotters,
        y_mean_plotters=y_mean_plotters,
        y_hat_plotters=y_hat_plotters,
        y_holdout_plotters=y_holdout_plotters,
        x_holdout_plotters=x_holdout_plotters,
        y_forecasts_plotters=y_forecasts_plotters,
        y_forecasts_mean_plotters=y_forecasts_mean_plotters,
        num_samples=num_samples,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        backend_kwargs=backend_kwargs,
        y_kwargs=y_kwargs,
        y_hat_plot_kwargs=y_hat_plot_kwargs,
        y_mean_plot_kwargs=y_mean_plot_kwargs,
        vline_kwargs=vline_kwargs,
        textsize=textsize,
        figsize=figsize,
        legend=legend,
        grid=grid,
        axes=axes,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_ts", "tsplot", backend)
    ax = plot(**tsplot_kwargs)
    return ax
