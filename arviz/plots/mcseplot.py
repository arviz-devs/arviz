"""Plot quantile MC standard error"""
import numpy as np
import xarray as xr

from ..data import convert_to_dataset
from ..stats import mcse
from .plot_utils import (
    xarray_var_iter,
    _scale_fig_size,
    make_label,
    default_grid,
    _create_axes_grid,
    get_coords,
)
from ..utils import _var_names


def plot_mcse(
    idata,
    var_names=None,
    coords=None,
    figsize=None,
    textsize=None,
    rug=False,
    rug_kind="diverging",
    n_points=20,
    ax=None,
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
    ax : axes, optional
        Matplotlib axes. Defaults to None.
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
            [
                mcse(data, var_names=var_names, method="quantile", prob=p)
                for p in probs
            ],
            dim="mcse_dim",
        )

    plotters = list(
        xarray_var_iter(mcse_dataset, var_names=var_names, skip_dims={"mcse_dim"})
    )
    length_plotters = len(plotters)
    rows, cols = default_grid(length_plotters)

    (figsize, ax_labelsize, titlesize, xt_labelsize, _linewidth, _markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )
    kwargs.setdefault("linestyle", kwargs.pop("ls", "none"))
    kwargs.setdefault("linewidth", kwargs.pop("lw", _linewidth))
    kwargs.setdefault("markersize", kwargs.pop("ms", _markersize))
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("zorder", 3)

    if ax is None:
        _, ax = _create_axes_grid(
            length_plotters, rows, cols, figsize=figsize, squeeze=False, constrained_layout=True
        )

    for (var_name, selection, x), ax_ in zip(plotters, np.ravel(ax)):
        ax_.plot(probs, x, **kwargs)
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

            values = data[var_name].sel(**selection).values.flatten()
            mask = idata.sample_stats[rug_kind].values.flatten()
            values = np.argsort(values)[mask]
            rug_space = np.max(x) * rug_kwargs.pop("space")
            rug_x, rug_y = values / (len(mask) - 1), np.zeros_like(values) - rug_space
            ax_.plot(rug_x, rug_y, **rug_kwargs)
            ax_.axhline(0, color="k", linewidth=_linewidth, alpha=0.7)

        ax_.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
        ax_.tick_params(labelsize=xt_labelsize)
        ax_.set_xlabel(
            "Quantile", fontsize=ax_labelsize, wrap=True
        )
        ax_.set_ylabel(
            "MCSE", fontsize=ax_labelsize, wrap=True
        )
        ax_.set_xlim(0, 1)
        if rug:
            ax_.yaxis.get_major_locator().set_params(nbins="auto", steps=[1, 2, 5, 10])
            _, ymax = ax_.get_ylim()
            yticks = ax_.get_yticks()
            yticks = yticks[(yticks >= 0) & (yticks < ymax)]
            ax_.set_yticks(yticks)
            ax_.set_yticklabels(["{:.3g}".format(ytick) for ytick in yticks])
        else:
            ax_.set_ylim(bottom=0)

    return ax
