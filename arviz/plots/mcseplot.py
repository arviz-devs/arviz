"""Plot quantile MC standard error."""
import numpy as np
import xarray as xr

from ..data import convert_to_dataset
from ..stats import mcse
from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    default_grid,
    get_coords,
    filter_plotters_list,
    get_plotting_function,
)
from ..utils import _var_names


def plot_mcse(
    idata,
    var_names=None,
    coords=None,
    errorbar=False,
    figsize=None,
    textsize=None,
    extra_methods=False,
    rug=False,
    rug_kind="diverging",
    n_points=20,
    ax=None,
    rug_kwargs=None,
    extra_kwargs=None,
    text_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
    **kwargs
):
    """Plot quantile or local Monte Carlo Standard Error.

    Parameters
    ----------
    idata : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names, optional
        Variables to be plotted.
    coords : dict, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    errorbar : bool, optional
        Plot quantile value +/- mcse instead of plotting mcse.
    figsize : tuple, optional
        Figure size. If None it will be defined automatically.
    textsize: float, optional
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    extra_methods : bool, optional
        Plot mean and sd MCSE as horizontal lines. Only taken into account when
        ``errorbar=False``.
    rug : bool
        Plot rug plot of values diverging or that reached the max tree depth.
    rug_kind : bool
        Variable in sample stats to use as rug mask. Must be a boolean variable.
    n_points : int
        Number of points for which to plot their quantile/local ess or number of subsets
        in the evolution plot.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    rug_kwargs : dict
        kwargs passed to rug plot.
    extra_kwargs : dict, optional
        kwargs passed to ax.plot for extra methods lines.
    text_kwargs : dict, optional
        kwargs passed to ax.annotate for extra methods lines labels. It accepts the additional
        key ``x`` to set ``xy=(text_kwargs["x"], mcse)``
    backend: str, optional
        Select plotting backend {"matplotlib","bokeh"}. Default "matplotlib".
    backend_kwargs: bool, optional
        These are kwargs specific to the backend being used. For additional documentation
        check the plotting method of the backend.
    show : bool, optional
        Call backend show function.
    **kwargs
        Passed as-is to plt.hist() or plt.plot() function depending on the value of `kind`.

    Returns
    -------
    axes : matplotlib axes or bokeh figures

    References
    ----------
    * Vehtari et al. (2019) see https://arxiv.org/abs/1903.08008

    Examples
    --------
    Plot quantile Monte Carlo Standard Error.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> idata = az.load_arviz_data("centered_eight")
        >>> coords = {"school": ["Deerfield", "Lawrenceville"]}
        >>> az.plot_mcse(
        ...     idata, var_names=["mu", "theta"], coords=coords
        ... )

    """
    mean_mcse = None
    sd_mcse = None
    text_x = None
    text_va = None

    if coords is None:
        coords = {}
    if "chain" in coords or "draw" in coords:
        raise ValueError("chain and draw are invalid coordinates for this kind of plot")

    data = get_coords(convert_to_dataset(idata, group="posterior"), coords)
    var_names = _var_names(var_names, data)

    probs = np.linspace(1 / n_points, 1 - 1 / n_points, n_points)
    mcse_dataset = xr.concat(
        [mcse(data, var_names=var_names, method="quantile", prob=p) for p in probs], dim="mcse_dim"
    )

    plotters = filter_plotters_list(
        list(xarray_var_iter(mcse_dataset, var_names=var_names, skip_dims={"mcse_dim"})),
        "plot_mcse",
    )
    length_plotters = len(plotters)
    rows, cols = default_grid(length_plotters)

    (figsize, ax_labelsize, titlesize, xt_labelsize, _linewidth, _markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )
    kwargs.setdefault("linestyle", kwargs.pop("ls", "none"))
    kwargs.setdefault("linewidth", kwargs.pop("lw", _linewidth))
    kwargs.setdefault("markersize", kwargs.pop("ms", _markersize))
    kwargs.setdefault("marker", "_" if errorbar else "o")
    kwargs.setdefault("zorder", 3)
    if extra_kwargs is None:
        extra_kwargs = {}
    extra_kwargs.setdefault("linestyle", extra_kwargs.pop("ls", "-"))
    extra_kwargs.setdefault("linewidth", extra_kwargs.pop("lw", _linewidth / 2))
    extra_kwargs.setdefault("color", "k")
    extra_kwargs.setdefault("alpha", 0.5)
    if extra_methods:
        mean_mcse = mcse(data, var_names=var_names, method="mean")
        sd_mcse = mcse(data, var_names=var_names, method="sd")
        if text_kwargs is None:
            text_kwargs = {}
        text_x = text_kwargs.pop("x", 1)
        text_kwargs.setdefault("fontsize", text_kwargs.pop("size", xt_labelsize * 0.7))
        text_kwargs.setdefault("alpha", extra_kwargs["alpha"])
        text_kwargs.setdefault("color", extra_kwargs["color"])
        text_kwargs.setdefault("horizontalalignment", text_kwargs.pop("ha", "right"))
        text_va = text_kwargs.pop("verticalalignment", text_kwargs.pop("va", None))

    mcse_kwargs = dict(
        ax=ax,
        plotters=plotters,
        length_plotters=length_plotters,
        rows=rows,
        cols=cols,
        figsize=figsize,
        errorbar=errorbar,
        rug=rug,
        data=data,
        probs=probs,
        kwargs=kwargs,
        extra_methods=extra_methods,
        mean_mcse=mean_mcse,
        sd_mcse=sd_mcse,
        text_x=text_x,
        text_va=text_va,
        text_kwargs=text_kwargs,
        rug_kwargs=rug_kwargs,
        extra_kwargs=extra_kwargs,
        idata=idata,
        rug_kind=rug_kind,
        _markersize=_markersize,
        _linewidth=_linewidth,
        xt_labelsize=xt_labelsize,
        ax_labelsize=ax_labelsize,
        titlesize=titlesize,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    if backend == "bokeh":
        mcse_kwargs.pop("kwargs")
        mcse_kwargs.pop("text_x")
        mcse_kwargs.pop("text_va")
        mcse_kwargs.pop("text_kwargs")
        mcse_kwargs.pop("xt_labelsize")
        mcse_kwargs.pop("ax_labelsize")
        mcse_kwargs.pop("titlesize")

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_mcse", "mcseplot", backend)
    ax = plot(**mcse_kwargs)
    return ax
