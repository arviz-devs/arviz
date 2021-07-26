"""Plot timeseries data."""
import numpy as np

from ..sel_utils import xarray_var_iter
from ..rcparams import rcParams
from .plot_utils import get_plotting_function


def plot_ts(
    idata,
    y=None,
    x=None,
    components=None,
    holdout=-1,
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
    idata : az.InferenceData object, Optional
        Optional only if y is Sequence
    y : str, variable name from observed_data
    x : str, Optional
        If none, coords name of y (y should be DataArray).
    components : list, variables from posterior, Optional
        Shape of the components must be like (chain, draw, *)
    holdout : int, Optional
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
    if isinstance(y, str):
        y_hat = idata.posterior_predictive[y]
        y = idata.observed_data[y]

    if isinstance(x, str):
        x = idata.constant_data[x]
    else:
        x = y.coords[y.dims[0]]

    components_plotters = []
    comp_uncertainty_plotters = []
    total_samples = y_hat.sizes["chain"] * y_hat.sizes["draw"]
    pp_sample_ix = np.random.choice(total_samples, size=num_samples, replace=False)

    if components is not None:
        for component in components:
            component_xr = idata.posterior[component]

            component_mean = component_xr.mean(("chain", "draw"))
            components_plotters += list(
                xarray_var_iter(
                    component_mean,
                    skip_dims=set(component_mean.dims),
                    combined=True,
                )
            )

            components_satcked = component_xr[..., -holdout:].stack(sample=("chain", "draw"))[
                ..., pp_sample_ix
            ]
            comp_uncertainty_plotters += list(
                xarray_var_iter(
                    components_satcked,
                    skip_dims=set(components_satcked.dims),
                    combined=True,
                )
            )

    y_plotters = list(
        xarray_var_iter(
            y,
            skip_dims=set(y.dims),
            combined=True,
        )
    )

    y_hat_satcked = y_hat[..., -holdout:].stack(sample=("chain", "draw"))[..., pp_sample_ix]
    y_hat_plotters = list(
        xarray_var_iter(
            y_hat_satcked,
            skip_dims=set(y_hat_satcked.dims),
            combined=True,
        )
    )

    y_hat_mean_plotters = y_hat.mean(("chain", "draw"))

    x_plotters = list(
        xarray_var_iter(
            x,
            skip_dims=set(x.dims),
            combined=True,
        )
    )

    rows = len(components_plotters) + len(y_plotters)
    cols = 1

    tsplot_kwargs = dict(
        x_plotters=x_plotters,
        y_plotters=y_plotters,
        y_hat_plotters=y_hat_plotters,
        y_hat_mean_plotters=y_hat_mean_plotters,
        components_plotters=components_plotters,
        comp_uncertainty_plotters=comp_uncertainty_plotters,
        holdout=holdout,
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
