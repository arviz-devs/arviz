"""Matplotlib distplot."""
import warnings
import matplotlib.pyplot as plt
import numpy as np
from . import backend_show
from ...kdeplot import plot_kde
from ...plot_utils import matplotlib_kwarg_dealiaser
from ....numeric_utils import get_bins


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
    is_circular,
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
        ax = plt.gca(polar=is_circular)

    if kind == "hist":
        ax = _histplot_mpl_op(
            values=values,
            values2=values2,
            rotated=rotated,
            ax=ax,
            hist_kwargs=hist_kwargs,
            is_circular=is_circular,
        )

    elif kind == "kde":
        plot_kwargs = matplotlib_kwarg_dealiaser(plot_kwargs, "plot")
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
            is_circular=is_circular,
            show=show,
        )

    if backend_show(show):
        plt.show()

    return ax


def _histplot_mpl_op(values, values2, rotated, ax, hist_kwargs, is_circular):
    """Add a histogram for the data to the axes."""
    bins = hist_kwargs.pop("bins")

    if is_circular == "degrees":
        if bins is None:
            bins = get_bins(values)
        values = np.deg2rad(values)
        bins = np.deg2rad(bins)
        ax.set_yticklabels([])

    elif is_circular == "radians":
        labels = [
            r"0",
            r"π/4",
            r"π/2",
            r"3π/4",
            r"π",
            r"5π/4",
            r"3π/2",
            r"7π/4",
        ]

        ax.set_xticklabels(labels)
        ax.set_yticklabels([])

    if values2 is not None:
        raise NotImplementedError("Insert hexbin plot here")

    if bins is None:
        bins = get_bins(values)

    n, _, _ = ax.hist(np.asarray(values).flatten(), bins=bins, **hist_kwargs)

    if rotated or is_circular:
        ax.set_yticks(bins[:-1])
    else:
        ax.set_xticks(bins[:-1])

    if is_circular:
        ax.set_ylim(0, n.max() + 0.5 * n.max())

    if hist_kwargs.get("label") is not None:
        ax.legend()

    return ax
