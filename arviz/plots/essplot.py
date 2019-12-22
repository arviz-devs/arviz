"""Plot quantile or local effective sample sizes."""
import numpy as np
import xarray as xr

from ..data import convert_to_dataset
from ..stats import ess
from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    default_grid,
    get_coords,
    filter_plotters_list,
    get_plotting_function,
)
from ..utils import _var_names


def plot_ess(
    idata,
    var_names=None,
    kind="local",
    relative=False,
    coords=None,
    figsize=None,
    textsize=None,
    rug=False,
    rug_kind="diverging",
    n_points=20,
    extra_methods=False,
    min_ess=400,
    ax=None,
    extra_kwargs=None,
    text_kwargs=None,
    hline_kwargs=None,
    rug_kwargs=None,
    backend=None,
    backend_kwargs=None,
    show=None,
    **kwargs
):
    """Plot quantile, local or evolution of effective sample sizes (ESS).

    Parameters
    ----------
    idata : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names, optional
        Variables to be plotted.
    kind : str, optional
        Options: ``local``, ``quantile`` or ``evolution``, specify the kind of plot.
    relative : bool
        Show relative ess in plot ``ress = ess / N``.
    coords : dict, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    figsize : tuple, optional
        Figure size. If None it will be defined automatically.
    textsize: float, optional
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    rug : bool
        Plot rug plot of values diverging or that reached the max tree depth.
    rug_kind : bool
        Variable in sample stats to use as rug mask. Must be a boolean variable.
    n_points : int
        Number of points for which to plot their quantile/local ess or number of subsets
        in the evolution plot.
    extra_methods : bool, optional
        Plot mean and sd ESS as horizontal lines. Not taken into account in evolution kind
    min_ess : int
        Minimum number of ESS desired.
    ax: axes, optional
        Matplotlib axes or bokeh figures.
    extra_kwargs : dict, optional
        If evolution plot, extra_kwargs is used to plot ess tail and differentiate it
        from ess bulk. Otherwise, passed to extra methods lines.
    text_kwargs : dict, optional
        Only taken into account when ``extra_methods=True``. kwargs passed to ax.annotate
        for extra methods lines labels. It accepts the additional
        key ``x`` to set ``xy=(text_kwargs["x"], mcse)``
    hline_kwargs : dict, optional
        kwargs passed to ax.axhline for the horizontal minimum ESS line.
    rug_kwargs : dict
        kwargs passed to rug plot.
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
    Plot local ESS. This plot, together with the quantile ESS plot, is recommended to check
    that there are enough samples for all the explored regions of parameter space. Checking
    local and quantile ESS is particularly relevant when working with credible intervals as
    opposed to ESS bulk, which is relevant for point estimates.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> idata = az.load_arviz_data("centered_eight")
        >>> coords = {"school": ["Choate", "Lawrenceville"]}
        >>> az.plot_ess(
        ...     idata, kind="local", var_names=["mu", "theta"], coords=coords
        ... )

    Plot quantile ESS.

    .. plot::
        :context: close-figs

        >>> az.plot_ess(
        ...     idata, kind="quantile", var_names=["mu", "theta"], coords=coords
        ... )

    Plot ESS evolution as the number of samples increase. When the model is converging properly,
    both lines in this plot should be roughly linear.

    .. plot::
        :context: close-figs

        >>> az.plot_ess(
        ...     idata, kind="evolution", var_names=["mu", "theta"], coords=coords
        ... )

    Customize local ESS plot to look like reference paper.

    .. plot::
        :context: close-figs

        >>> az.plot_ess(
        ...     idata, kind="local", var_names=["mu"], drawstyle="steps-mid", color="k",
        ...     linestyle="-", marker=None, rug=True, rug_kwargs={"color": "r"}
        ... )

    Customize ESS evolution plot to look like reference paper.

    .. plot::
        :context: close-figs

        >>> extra_kwargs = {"color": "lightsteelblue"}
        >>> az.plot_ess(
        ...     idata, kind="evolution", var_names=["mu"],
        ...     color="royalblue", extra_kwargs=extra_kwargs
        ... )

    """
    valid_kinds = ("local", "quantile", "evolution")
    kind = kind.lower()
    if kind not in valid_kinds:
        raise ValueError("Invalid kind, kind must be one of {} not {}".format(valid_kinds, kind))

    if coords is None:
        coords = {}
    if "chain" in coords or "draw" in coords:
        raise ValueError("chain and draw are invalid coordinates for this kind of plot")
    extra_methods = False if kind == "evolution" else extra_methods

    data = get_coords(convert_to_dataset(idata, group="posterior"), coords)
    var_names = _var_names(var_names, data)
    n_draws = data.dims["draw"]
    n_samples = n_draws * data.dims["chain"]

    ess_tail_dataset = None
    mean_ess = None
    sd_ess = None
    text_x = None
    text_va = None

    if kind == "quantile":
        probs = np.linspace(1 / n_points, 1 - 1 / n_points, n_points)
        xdata = probs
        ylabel = "{} for quantiles"
        ess_dataset = xr.concat(
            [
                ess(data, var_names=var_names, relative=relative, method="quantile", prob=p)
                for p in probs
            ],
            dim="ess_dim",
        )
    elif kind == "local":
        probs = np.linspace(0, 1, n_points, endpoint=False)
        xdata = probs
        ylabel = "{} for small intervals"
        ess_dataset = xr.concat(
            [
                ess(
                    data,
                    var_names=var_names,
                    relative=relative,
                    method="local",
                    prob=[p, p + 1 / n_points],
                )
                for p in probs
            ],
            dim="ess_dim",
        )
    else:
        first_draw = data.draw.values[0]
        ylabel = "{}"
        xdata = np.linspace(n_samples / n_points, n_samples, n_points)
        draw_divisions = np.linspace(n_draws // n_points, n_draws, n_points, dtype=int)
        ess_dataset = xr.concat(
            [
                ess(
                    data.sel(draw=slice(first_draw + draw_div)),
                    var_names=var_names,
                    relative=relative,
                    method="bulk",
                )
                for draw_div in draw_divisions
            ],
            dim="ess_dim",
        )
        ess_tail_dataset = xr.concat(
            [
                ess(
                    data.sel(draw=slice(first_draw + draw_div)),
                    var_names=var_names,
                    relative=relative,
                    method="tail",
                )
                for draw_div in draw_divisions
            ],
            dim="ess_dim",
        )

    plotters = filter_plotters_list(
        list(xarray_var_iter(ess_dataset, var_names=var_names, skip_dims={"ess_dim"})), "plot_ess"
    )
    length_plotters = len(plotters)
    rows, cols = default_grid(length_plotters)

    (figsize, ax_labelsize, titlesize, xt_labelsize, _linewidth, _markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )
    _linestyle = kwargs.pop("ls", "-" if kind == "evolution" else "none")
    kwargs.setdefault("linestyle", _linestyle)
    kwargs.setdefault("linewidth", kwargs.pop("lw", _linewidth))
    kwargs.setdefault("markersize", kwargs.pop("ms", _markersize))
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("zorder", 3)
    if extra_kwargs is None:
        extra_kwargs = {}
    if kind == "evolution":
        extra_kwargs = {
            **extra_kwargs,
            **{key: item for key, item in kwargs.items() if key not in extra_kwargs},
        }
        kwargs.setdefault("label", "bulk")
        extra_kwargs.setdefault("label", "tail")
    else:
        extra_kwargs.setdefault("linestyle", extra_kwargs.pop("ls", "-"))
        extra_kwargs.setdefault("linewidth", extra_kwargs.pop("lw", _linewidth / 2))
        extra_kwargs.setdefault("color", "k")
        extra_kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("label", kind)
    if hline_kwargs is None:
        hline_kwargs = {}
    hline_kwargs.setdefault("linewidth", hline_kwargs.pop("lw", _linewidth))
    hline_kwargs.setdefault("linestyle", hline_kwargs.pop("ls", "--"))
    hline_kwargs.setdefault("color", hline_kwargs.pop("c", "gray"))
    hline_kwargs.setdefault("alpha", 0.7)
    if extra_methods:
        mean_ess = ess(data, var_names=var_names, method="mean", relative=relative)
        sd_ess = ess(data, var_names=var_names, method="sd", relative=relative)
        if text_kwargs is None:
            text_kwargs = {}
        text_x = text_kwargs.pop("x", 1)
        text_kwargs.setdefault("fontsize", text_kwargs.pop("size", xt_labelsize * 0.7))
        text_kwargs.setdefault("alpha", extra_kwargs["alpha"])
        text_kwargs.setdefault("color", extra_kwargs["color"])
        text_kwargs.setdefault("horizontalalignment", text_kwargs.pop("ha", "right"))
        text_va = text_kwargs.pop("verticalalignment", text_kwargs.pop("va", None))

    essplot_kwargs = dict(
        ax=ax,
        plotters=plotters,
        xdata=xdata,
        ess_tail_dataset=ess_tail_dataset,
        mean_ess=mean_ess,
        sd_ess=sd_ess,
        idata=idata,
        data=data,
        text_x=text_x,
        text_va=text_va,
        kind=kind,
        extra_methods=extra_methods,
        rows=rows,
        cols=cols,
        figsize=figsize,
        kwargs=kwargs,
        extra_kwargs=extra_kwargs,
        text_kwargs=text_kwargs,
        _linewidth=_linewidth,
        _markersize=_markersize,
        n_samples=n_samples,
        relative=relative,
        min_ess=min_ess,
        xt_labelsize=xt_labelsize,
        titlesize=titlesize,
        ax_labelsize=ax_labelsize,
        ylabel=ylabel,
        rug=rug,
        rug_kind=rug_kind,
        rug_kwargs=rug_kwargs,
        hline_kwargs=hline_kwargs,
        backend_kwargs=backend_kwargs,
        show=show,
    )

    # TODO: Add backend kwargs
    plot = get_plotting_function("plot_ess", "essplot", backend)
    ax = plot(**essplot_kwargs)
    return ax
