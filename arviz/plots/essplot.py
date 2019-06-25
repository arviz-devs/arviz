"""Plot quantile or local effective sample sizes."""
import numpy as np
import xarray as xr

from ..data import convert_to_dataset
from ..stats import ess
from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    make_label,
    default_grid,
    _create_axes_grid,
    get_coords,
)
from ..utils import _var_names


def plot_ess(
    data,
    var_names=None,
    kind="local",
    relative=False,
    coords=None,
    figsize=None,
    textsize=None,
    n_points=20,
    ax=None,
    extra_kwargs=None,
    **kwargs
):
    """Plot quantile, local or change of effective sample sizes (ESS).

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names, optional
        Variables to be plotted, two variables are required.
    kind : str, optional
        Options: ``local``, ``quantile`` or ``change``, specify the kind of plot.
    relative : bool
        Show relative ess in plot.
    coords : dict, optional
        Coordinates of var_names to be plotted. Passed to `Dataset.sel`
    figsize : tuple, optional
        Figure size. If None it will be defined automatically.
    textsize: float, optional
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    n_points : int
        Number of quantiles for which to plot their quantile/local ess or number of subsets
        in the change plot.
    ax : axes, optional
        Matplotlib axes. Defaults to None.
    extra_kwargs : dict
        kwargs used to plot ess tail and differentiate it from ess bulk. If None, the same
        kwargs are used, thus, the 2 lines will differ in the color which is matplotlib default.
    **kwargs
        Passed as-is to plt.hist() or plt.plot() function depending on the value of `kind`.

    Returns
    -------
    ax : matplotlib axes

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

    Plot ESS change as the number of samples increase. When the model is converging properly,
    both lines in this plot should be roughly linear.

    .. plot::
        :context: close-figs

        >>> az.plot_ess(
        ...     idata, kind="change", var_names=["mu", "theta"], coords=coords
        ... )

    Customize local ESS plot to look like reference paper.

    .. plot::
        :context: close-figs

        >>> az.plot_ess(
        ...     idata, kind="local", var_names=["mu"], drawstyle="steps-mid",
        ...     color="k", linestyle="-", marker=None
        ... )

    Customize ESS change plot to look like reference paper.

    .. plot::
        :context: close-figs

        >>> extra_kwargs = {"color": "lightsteelblue"}
        >>> az.plot_ess(
        ...     idata, kind="change", var_names=["mu"], color="royalblue", extra_kwargs=extra_kwargs
        ... )

    """
    valid_kinds = ("local", "quantile", "change")
    kind = kind.lower()
    if kind not in valid_kinds:
        raise ValueError("Invalid kind, kind must be one of {} not {}".format(valid_kinds, kind))

    if coords is None:
        coords = {}

    data = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, data)

    if kind == "quantile":
        probs = np.arange(1 / n_points, 1, 1 / n_points)
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
        probs = np.arange(0, 1, 1 / n_points)
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
        n_draws = len(data.draw)
        n_samples = n_draws * len(data.chain)
        ylabel = "{}"
        xdata = np.linspace(n_samples / n_points, n_samples, n_points)
        draw_divisions = np.linspace(n_draws / n_points, n_draws, n_points)
        ess_dataset = xr.concat(
            [
                ess(
                    data.sel(draw=slice(draw_div)),
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
                    data.sel(draw=slice(draw_div)),
                    var_names=var_names,
                    relative=relative,
                    method="tail",
                )
                for draw_div in draw_divisions
            ],
            dim="ess_dim",
        )

    plotters = list(
        xarray_var_iter(get_coords(ess_dataset, coords), var_names=var_names, skip_dims={"ess_dim"})
    )
    length_plotters = len(plotters)
    rows, cols = default_grid(length_plotters)

    (figsize, ax_labelsize, titlesize, xt_labelsize, _linewidth, _markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )
    kwargs.setdefault("linestyle", "-" if kind == "change" else "none")
    kwargs.setdefault("linewidth", _linewidth)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", _markersize)
    if kind == "change":
        if extra_kwargs is None:
            extra_kwargs = {}
        extra_kwargs = {
            **extra_kwargs,
            **{key: item for key, item in kwargs.items() if key not in extra_kwargs},
        }
        kwargs.setdefault("label", "bulk")
        extra_kwargs.setdefault("label", "tail")

    if ax is None:
        _, ax = _create_axes_grid(
            length_plotters, rows, cols, figsize=figsize, squeeze=False, constrained_layout=True
        )

    for (var_name, selection, x), ax_ in zip(plotters, np.ravel(ax)):
        ax_.plot(xdata, x, **kwargs)
        if kind == "change":
            ess_tail = ess_tail_dataset[var_name].sel(**selection)
            ax_.plot(xdata, ess_tail, **extra_kwargs)
        ax_.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
        ax_.tick_params(labelsize=xt_labelsize)
        ax_.set_xlabel(
            "Total number of draws" if kind == "change" else "Quantile", fontsize=ax_labelsize
        )
        ax_.set_ylabel(
            ylabel.format("Relative ESS" if relative else "ESS"), fontsize=ax_labelsize, wrap=True
        )
        if kind == "change":
            ax_.legend(title="type")
        else:
            ax_.set_xlim(0, 1)

    return ax
