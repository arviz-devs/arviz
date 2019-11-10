"""Matplotlib distplot"""
from ...kdeplot import plot_kde
import matplotlib.pyplot as plt


def _plot_dist_mpl(
    values,
    values2=None,
    color="C0",
    kind="auto",
    cumulative=False,
    label=None,
    rotated=False,
    rug=False,
    bw=4.5,
    quantiles=None,
    contour=True,
    fill_last=True,
    textsize=None,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    contour_kwargs=None,
    hist_kwargs=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    if kind == "hist":
        ax = _histplot_mpl_op(
            values=values, values2=values2, rotated=rotated, ax=ax, hist_kwargs=hist_kwargs
        )

    elif kind == "density":
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
            ax=ax,
            backend="matplotlib",
        )

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
