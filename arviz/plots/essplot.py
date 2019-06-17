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
    **kwargs
):
    """Plot quantile or local effective sample sizes.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an az.InferenceData object
        Refer to documentation of az.convert_to_dataset for details
    var_names : list of variable names, optional
        Variables to be plotted, two variables are required.
    kind : str, optional
        Options: ``local`` or ``quantile``, whether to plot local or quantile ess plot.
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
        Number of quantiles for which to plot their quantile/local ess
    ax : axes, optional
        Matplotlib axes. Defaults to None.
    **kwargs
        Passed as-is to plt.hist() or plt.plot() function depending on the value of `kind`.

    Returns
    -------
    ax : matplotlib axes

    Examples
    --------
    """
    valid_kinds = ("local", "quantile")
    kind = kind.lower()
    if kind not in valid_kinds:
        raise ValueError("Invalid kind, kind must be one of {} not {}".format(valid_kinds, kind))

    if coords is None:
        coords = {}

    data = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, data)

    if kind == "quantile":
        probs = np.arange(1 / n_points, 1, 1 / n_points)
        ess_dataset = xr.concat(
            [
                ess(data, var_names=var_names, relative=relative, method="quantile", prob=p)
                for p in probs
            ],
            dim="quantile",
        )
    elif kind == "local":
        probs = np.arange(0, 1, 1 / n_points)
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
            dim="quantile",
        )

    plotters = list(
        xarray_var_iter(
            get_coords(ess_dataset, coords), var_names=var_names, skip_dims={"quantile"}
        )
    )
    length_plotters = len(plotters)
    rows, cols = default_grid(length_plotters)

    (figsize, ax_labelsize, titlesize, xt_labelsize, _linewidth, _markersize) = _scale_fig_size(
        figsize, textsize, rows, cols
    )
    kwargs.setdefault("linestyle", "none")
    kwargs.setdefault("linewidth", _linewidth)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", _markersize)

    if ax is None:
        _, ax = _create_axes_grid(
            length_plotters, rows, cols, figsize=figsize, squeeze=False, constrained_layout=True
        )

    for (var_name, selection, x), ax_ in zip(plotters, np.ravel(ax)):
        ax_.plot(probs, x, **kwargs)
        ax_.set_title(make_label(var_name, selection), fontsize=titlesize, wrap=True)
        ax_.tick_params(labelsize=xt_labelsize)
        ax_.set_xlabel("Quantile", fontsize=ax_labelsize)
        ax_.set_ylabel(
            "{} for {}".format(
                "Relative ESS" if relative else "ESS",
                "small intervals" if kind == "local" else "quantiles",
            ),
            fontsize=ax_labelsize,
            wrap=True,
        )
        ax_.set_xlim(0, 1)

    return ax
