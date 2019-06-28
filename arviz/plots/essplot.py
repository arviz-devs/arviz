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
    min_ess=400,
    ax=None,
    extra_kwargs=None,
    hline_kwargs=None,
    rug_kwargs=None,
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
    min_ess : int
        Minimum number of ESS desired.
    ax : axes, optional
        Matplotlib axes. Defaults to None.
    extra_kwargs : dict
        kwargs used to plot ess tail and differentiate it from ess bulk. If None, the same
        kwargs are used, thus, the 2 lines will differ in the color which is matplotlib default.
    hline_kwargs : dict
        kwargs passed to ax.axhline for the horizontal minimum ESS line.
    rug_kwargs : dict
        kwargs passed to rug plot.
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

    data = convert_to_dataset(idata, group="posterior")
    var_names = _var_names(var_names, data)
    n_draws = data.dims["draw"]
    n_samples = n_draws * data.dims["chain"]

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

    plotters = list(
        xarray_var_iter(get_coords(ess_dataset, coords), var_names=var_names, skip_dims={"ess_dim"})
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
    if kind == "evolution":
        if extra_kwargs is None:
            extra_kwargs = {}
        extra_kwargs = {
            **extra_kwargs,
            **{key: item for key, item in kwargs.items() if key not in extra_kwargs},
        }
        kwargs.setdefault("label", "bulk")
        extra_kwargs.setdefault("label", "tail")
    if hline_kwargs is None:
        hline_kwargs = {}
    hline_kwargs.setdefault("linewidth", hline_kwargs.pop("lw", _linewidth))
    hline_kwargs.setdefault("linestyle", hline_kwargs.pop("ls", "--"))
    hline_kwargs.setdefault("color", hline_kwargs.pop("c", "gray"))
    hline_kwargs.setdefault("alpha", 0.7)

    if ax is None:
        _, ax = _create_axes_grid(
            length_plotters, rows, cols, figsize=figsize, squeeze=False, constrained_layout=True
        )

    for (var_name, selection, x), ax_ in zip(plotters, np.ravel(ax)):
        ax_.plot(xdata, x, **kwargs)
        if kind == "evolution":
            ess_tail = ess_tail_dataset[var_name].sel(**selection)
            ax_.plot(xdata, ess_tail, **extra_kwargs)
        elif rug:
            if rug_kwargs is None:
                rug_kwargs = {}
            if not hasattr(idata, "sample_stats"):
                raise ValueError("InferenceData object must contain sample_stats for rug plot")
            if not hasattr(idata.sample_stats, rug_kind):
                raise ValueError("InferenceData does not contain {} data".format(rug_kind))
            rug_kwargs.setdefault("marker", "|")
            rug_kwargs.setdefault("linestyle", rug_kwargs.pop("ls", "None"))
            rug_kwargs.setdefault("color", rug_kwargs.pop("c", kwargs.get("color", "C0")))
            rug_kwargs.setdefault("space", 0.1)
            rug_kwargs.setdefault("markersize", rug_kwargs.pop("ms", 2 * _markersize))

            values = data[var_name].sel(**selection).values.flatten()
            mask = idata.sample_stats[rug_kind].values.flatten()
            values = np.argsort(values)[mask]
            rug_space = np.max(x) * rug_kwargs.pop("space")
            rug_x, rug_y = values / (len(mask) - 1), np.zeros_like(values) - rug_space
            ax_.plot(rug_x, rug_y, **rug_kwargs)
            ax_.axhline(0, color="k", linewidth=_linewidth, alpha=0.7)

        ax_.axhline(400 / n_samples if relative else min_ess, **hline_kwargs)

        ax_.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
        ax_.tick_params(labelsize=xt_labelsize)
        ax_.set_xlabel(
            "Total number of draws" if kind == "evolution" else "Quantile", fontsize=ax_labelsize
        )
        ax_.set_ylabel(
            ylabel.format("Relative ESS" if relative else "ESS"), fontsize=ax_labelsize, wrap=True
        )
        if kind == "evolution":
            ax_.legend(title="type")
        else:
            ax_.set_xlim(0, 1)
        if rug:
            ax_.yaxis.get_major_locator().set_params(nbins="auto", steps=[1, 2, 5, 10])
            _, ymax = ax_.get_ylim()
            yticks = ax_.get_yticks().astype(np.int64)
            yticks = yticks[(yticks >= 0) & (yticks < ymax)]
            ax_.set_yticks(yticks)
            ax_.set_yticklabels(yticks)
        else:
            ax_.set_ylim(bottom=0)

    return ax
