"""Plot quantile MC standard error."""
import numpy as np
import xarray as xr
from scipy.stats import rankdata

from ..data import convert_to_dataset
from ..stats import mcse
from ..stats.stats_utils import quantile as _quantile
from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    make_label,
    default_grid,
    _create_axes_grid,
    get_coords,
    filter_plotters_list,
)
from ..utils import _var_names


def plot_mcse(
    # disable black until #763 is released
    # fmt: off
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
    **kwargs
    # fmt: on
):
    """Plot quantile, local or evolution of effective sample sizes (ESS).

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
    ax : axes, optional
        Matplotlib axes. Defaults to None.
    rug_kwargs : dict
        kwargs passed to rug plot.
    extra_kwargs : dict, optional
        kwargs passed to ax.plot for extra methods lines.
    text_kwargs : dict, optional
        kwargs passed to ax.annotate for extra methods lines labels. It accepts the additional
        key ``x`` to set ``xy=(text_kwargs["x"], mcse)``
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
    Plot quantile MCSE.

    .. plot::
        :context: close-figs

        >>> import arviz as az
        >>> idata = az.load_arviz_data("centered_eight")
        >>> coords = {"school": ["Deerfield", "Lawrenceville"]}
        >>> az.plot_mcse(
        ...     idata, var_names=["mu", "theta"], coords=coords
        ... )

    """
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

    if ax is None:
        _, ax = _create_axes_grid(
            length_plotters, rows, cols, figsize=figsize, squeeze=False, constrained_layout=True
        )

    for (var_name, selection, x), ax_ in zip(plotters, np.ravel(ax)):
        if errorbar or rug:
            values = data[var_name].sel(**selection).values.flatten()
        if errorbar:
            quantile_values = _quantile(values, probs)
            ax_.errorbar(probs, quantile_values, yerr=x, **kwargs)
        else:
            ax_.plot(probs, x, label="quantile", **kwargs)
            if extra_methods:
                mean_mcse_i = mean_mcse[var_name].sel(**selection).values.item()
                sd_mcse_i = sd_mcse[var_name].sel(**selection).values.item()
                ax_.axhline(mean_mcse_i, **extra_kwargs)
                ax_.annotate(
                    "mean",
                    (text_x, mean_mcse_i),
                    va=text_va
                    if text_va is not None
                    else "bottom"
                    if mean_mcse_i > sd_mcse_i
                    else "top",
                    **text_kwargs,
                )
                ax_.axhline(sd_mcse_i, **extra_kwargs)
                ax_.annotate(
                    "sd",
                    (text_x, sd_mcse_i),
                    va=text_va
                    if text_va is not None
                    else "bottom"
                    if sd_mcse_i >= mean_mcse_i
                    else "top",
                    **text_kwargs,
                )
        if rug:
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

            mask = idata.sample_stats[rug_kind].values.flatten()
            values = rankdata(values)[mask]
            y_min, y_max = ax_.get_ylim()
            y_min = y_min if errorbar else 0
            rug_space = (y_max - y_min) * rug_kwargs.pop("space")
            rug_x, rug_y = values / (len(mask) - 1), np.full_like(values, y_min) - rug_space
            ax_.plot(rug_x, rug_y, **rug_kwargs)
            ax_.axhline(y_min, color="k", linewidth=_linewidth, alpha=0.7)

        ax_.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
        ax_.tick_params(labelsize=xt_labelsize)
        ax_.set_xlabel("Quantile", fontsize=ax_labelsize, wrap=True)
        ax_.set_ylabel(
            r"Value $\pm$ MCSE for quantiles" if errorbar else "MCSE for quantiles",
            fontsize=ax_labelsize,
            wrap=True,
        )
        ax_.set_xlim(0, 1)
        if rug:
            ax_.yaxis.get_major_locator().set_params(nbins="auto", steps=[1, 2, 5, 10])
            y_min, y_max = ax_.get_ylim()
            yticks = ax_.get_yticks()
            yticks = yticks[(yticks >= y_min) & (yticks < y_max)]
            ax_.set_yticks(yticks)
            ax_.set_yticklabels(["{:.3g}".format(ytick) for ytick in yticks])
        elif not errorbar:
            ax_.set_ylim(bottom=0)

    return ax
