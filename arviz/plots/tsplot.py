"""Plot timeseries data."""
import numpy as np

from ..sel_utils import xarray_var_iter
from ..rcparams import rcParams
from .plot_utils import get_plotting_function


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
    textsize=None,
    figsize=None,
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
    if y_hat is None and isinstance(y, str):
        y_hat = y

    if y_forecasts is None and isinstance(y_holdout, str):
        y_forecasts = y_holdout

    if isinstance(y, str):
        y = idata.observed_data[y]

    if len(y.dims) > 1 and plot_dim is None:
        raise ValueError("Argument plot_dim is needed in case of multidimensional data")

    if isinstance(x, str):
        x = idata.constant_data[x]
    elif x is None:
        if plot_dim is None:
            x = y.coords[y.dims[0]]
        else:
            x = y.coords[plot_dim]
    else:
        TypeError("Invalid datatype for x")

    if isinstance(y_hat, str):
        y_hat = idata.posterior_predictive[y_hat]

    if isinstance(y_forecasts, str):
        y_forecasts = idata.posterior_predictive[y_forecasts]
        if x_holdout is None:
            if holdout_dim is None:
                x_holdout = y_forecasts.coords[y_forecasts.dims[-1]]
            else:
                x_holdout = y_forecasts.coords[holdout_dim]
        elif isinstance(x_holdout, str):
            x_holdout = idata.constant_data[x_holdout]

    if isinstance(y_holdout, str):
        y_holdout = idata.observed_data[y_holdout]
        if len(y.dims) > 1 and holdout_dim is None:
            raise ValueError("Argument holdout_dim is needed in case of multidimentional data")
        if x_holdout is None:
            if holdout_dim is None:
                x_holdout = y_holdout.coords[y_forecasts.dims[-1]]
            else:
                x_holdout = y_holdout.coords[holdout_dim]
        elif isinstance(x_holdout, str):
            x_holdout = idata.constant_data[x_holdout]

    if plot_dim is None:
        skip_dims = list(y.dims)
    elif isinstance(plot_dim, str):
        skip_dims = [plot_dim]
    elif isinstance(plot_dim, tuple):
        skip_dims = list(plot_dim)

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
            skip_dims=set(x.dims),
            combined=True,
        )
    )
    x_plotters = np.tile(x_plotters, (len(y_plotters), 1))

    y_mean_plotters = None
    y_uncertainty_plotters = None
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
        y_uncertainty_plotters = y_hat_plotters

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
                skip_dims=set(x_holdout.dims),
                combined=True,
            )
        )
        x_holdout_plotters = np.tile(x_holdout_plotters, (len(y_holdout_plotters), 1))

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
                skip_dims=set(x_holdout.dims),
                combined=True,
            )
        )
        x_holdout_plotters = np.tile(x_holdout_plotters, (len(y_forecasts_plotters), 1))

    rows = len(y_plotters)
    cols = 1

    tsplot_kwargs = dict(
        x_plotters=x_plotters,
        y_plotters=y_plotters,
        y_mean_plotters=y_mean_plotters,
        y_hat_plotters=y_uncertainty_plotters,
        y_holdout_plotters=y_holdout_plotters,
        x_holdout_plotters=x_holdout_plotters,
        y_forecasts_plotters=y_forecasts_plotters,
        y_forecasts_mean_plotters=y_forecasts_mean_plotters,
        num_samples=num_samples,
        rows=rows,
        cols=cols,
        backend_kwargs=backend_kwargs,
        textsize=textsize,
        figsize=figsize,
        axes=axes,
        show=show,
    )

    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    plot = get_plotting_function("plot_ts", "tsplot", backend)
    ax = plot(**tsplot_kwargs)
    return ax
