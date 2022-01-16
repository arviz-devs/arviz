"""Matplotlib distplot."""
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers
import numpy as np

from ....stats.density_utils import get_bins
from ...kdeplot import plot_kde
from ...plot_utils import _scale_fig_size, _init_kwargs_dict
from . import backend_kwarg_defaults, backend_show, create_axes_grid, matplotlib_kwarg_dealiaser


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
    figsize,
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
    backend_kwargs = _init_kwargs_dict(backend_kwargs)

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, *_ = _scale_fig_size(figsize, textsize)

    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True
    backend_kwargs.setdefault("subplot_kw", {})
    backend_kwargs["subplot_kw"].setdefault("polar", is_circular)

    if ax is None:
        fig_manager = _pylab_helpers.Gcf.get_active()
        if fig_manager is not None:
            ax = fig_manager.canvas.figure.gca()
        else:
            _, ax = create_axes_grid(
                1,
                backend_kwargs=backend_kwargs,
            )

    if kind == "hist":
        hist_kwargs = matplotlib_kwarg_dealiaser(hist_kwargs, "hist")
        hist_kwargs.setdefault("cumulative", cumulative)
        hist_kwargs.setdefault("color", color)
        hist_kwargs.setdefault("label", label)
        hist_kwargs.setdefault("rwidth", 0.9)
        hist_kwargs.setdefault("align", "left")
        hist_kwargs.setdefault("density", True)

        if rotated:
            hist_kwargs.setdefault("orientation", "horizontal")
        else:
            hist_kwargs.setdefault("orientation", "vertical")

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
    bins = hist_kwargs.pop("bins", None)

    if is_circular == "degrees":
        if bins is None:
            bins = get_bins(values)
        values = np.deg2rad(values)
        bins = np.deg2rad(bins)

    elif is_circular:
        labels = [
            "0",
            f"{np.pi/4:.2f}",
            f"{np.pi/2:.2f}",
            f"{3*np.pi/4:.2f}",
            f"{np.pi:.2f}",
            f"{-3*np.pi/4:.2f}",
            f"{-np.pi/2:.2f}",
            f"{-np.pi/4:.2f}",
        ]

        ax.set_xticklabels(labels)

    if values2 is not None:
        raise NotImplementedError("Insert hexbin plot here")

    if bins is None:
        bins = get_bins(values)

    n, bins, _ = ax.hist(np.asarray(values).flatten(), bins=bins, **hist_kwargs)

    if rotated:
        ax.set_yticks(bins[:-1])
    elif not is_circular:
        ax.set_xticks(bins[:-1])

    if is_circular:
        ax.set_ylim(0, 1.5 * n.max())
        ax.set_yticklabels([])

    if hist_kwargs.get("label") is not None:
        ax.legend()

    return ax
