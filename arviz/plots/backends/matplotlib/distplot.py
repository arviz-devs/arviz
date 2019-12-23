"""Matplotlib distplot."""
import warnings
import matplotlib.pyplot as plt

from . import backend_show
from ...kdeplot import plot_kde


def plot_dist(
    values,
    values2,
    color,
    kind,
    cumulative,
    label,
    rotated,
    rug,
    bw,
    quantiles,
    contour,
    fill_last,
    textsize,
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
    contour_kwargs,
    contourf_kwargs,
    pcolormesh_kwargs,
    hist_kwargs,
    ax,
    backend_kwargs,
    show,
):
    """Matplotlib distplot."""
    if backend_kwargs is not None:
        warnings.warn(
            (
                "Argument backend_kwargs has not effect in matplotlib.plot_dist"
                "Supplied value won't be used"
            )
        )
        backend_kwargs = None
    if ax is None:
        ax = plt.gca()

    if kind == "hist":
        ax = _histplot_mpl_op(
            values=values, values2=values2, rotated=rotated, ax=ax, hist_kwargs=hist_kwargs
        )

    elif kind == "kde":
        if plot_kwargs is None:
            plot_kwargs = {}

        plot_kwargs.setdefault("color", color)
        legend = label is not None

        ax = plot_kde(
            values,
            values2,
            cumulative=cumulative,
            rug=rug,
            label=label,
            bw=bw,
            quantiles=quantiles,
            rotated=rotated,
            contour=contour,
            legend=legend,
            fill_last=fill_last,
            textsize=textsize,
            plot_kwargs=plot_kwargs,
            fill_kwargs=fill_kwargs,
            rug_kwargs=rug_kwargs,
            contour_kwargs=contour_kwargs,
            contourf_kwargs=contourf_kwargs,
            pcolormesh_kwargs=pcolormesh_kwargs,
            ax=ax,
            backend="matplotlib",
            backend_kwargs=backend_kwargs,
            show=show,
        )

    if backend_show(show):
        plt.show()

    return ax


def _histplot_mpl_op(values, values2, rotated, ax, hist_kwargs):
    """Add a histogram for the data to the axes."""
    if values2 is not None:
        raise NotImplementedError("Insert hexbin plot here")

    bins = hist_kwargs.pop("bins")

    ax.hist(values, bins=bins, **hist_kwargs)
    if rotated:
        ax.set_yticks(bins[:-1])
    else:
        ax.set_xticks(bins[:-1])
    if hist_kwargs.get("label") is not None:
        ax.legend()
    return ax
