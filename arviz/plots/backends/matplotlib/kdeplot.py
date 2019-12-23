"""Matplotlib kdeplot."""
import warnings
from matplotlib import pyplot as plt
import numpy as np

from . import backend_show
from ...plot_utils import _scale_fig_size


def plot_kde(
    density,
    lower,
    upper,
    density_q,
    xmin,
    xmax,
    ymin,
    ymax,
    gridsize,
    values,
    values2,
    rug,
    label,
    quantiles,
    rotated,
    contour,
    fill_last,
    textsize,
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
    contour_kwargs,
    contourf_kwargs,
    pcolormesh_kwargs,
    ax,
    legend,
    backend_kwargs,
    show,
):
    """Matplotlib kde plot."""
    if backend_kwargs is not None:
        warnings.warn(
            (
                "Argument backend_kwargs has not effect in matplotlib.plot_kde"
                "Supplied value won't be used"
            )
        )
    if ax is None:
        ax = plt.gca()

    figsize = ax.get_figure().get_size_inches()

    figsize, *_, xt_labelsize, linewidth, markersize = _scale_fig_size(figsize, textsize, 1, 1)

    if values2 is None:
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.setdefault("color", "C0")

        default_color = plot_kwargs.get("color")

        if fill_kwargs is None:
            fill_kwargs = {}

        fill_kwargs.setdefault("color", default_color)

        if rug_kwargs is None:
            rug_kwargs = {}
        rug_kwargs.setdefault("marker", "_" if rotated else "|")
        rug_kwargs.setdefault("linestyle", "None")
        rug_kwargs.setdefault("color", default_color)
        rug_kwargs.setdefault("space", 0.2)

        plot_kwargs.setdefault("linewidth", linewidth)
        rug_kwargs.setdefault("markersize", 2 * markersize)

        figsize = ax.get_figure().get_size_inches()

        figsize, *_, xt_labelsize, linewidth, markersize = _scale_fig_size(figsize, textsize, 1, 1)

        rug_space = max(density) * rug_kwargs.pop("space")

        x = np.linspace(lower, upper, len(density))

        fill_func = ax.fill_between
        fill_x, fill_y = x, density
        if rotated:
            x, density = density, x
            fill_func = ax.fill_betweenx

        ax.tick_params(labelsize=xt_labelsize)

        if rotated:
            ax.set_xlim(0, auto=True)
            rug_x, rug_y = np.zeros_like(values) - rug_space, values
        else:
            ax.set_ylim(0, auto=True)
            rug_x, rug_y = values, np.zeros_like(values) - rug_space

        if rug:
            ax.plot(rug_x, rug_y, **rug_kwargs)

        if quantiles is not None:
            fill_kwargs.setdefault("alpha", 0.75)

            idx = [np.sum(density_q < quant) for quant in quantiles]

            fill_func(
                fill_x,
                fill_y,
                where=np.isin(fill_x, fill_x[idx], invert=True, assume_unique=True),
                **fill_kwargs
            )
        else:
            fill_kwargs.setdefault("alpha", 0)
            if fill_kwargs.get("alpha") == 0:
                ax.plot(x, density, label=label, **plot_kwargs)
                fill_func(fill_x, fill_y, **fill_kwargs)
            else:
                ax.plot(x, density, **plot_kwargs)
                fill_func(fill_x, fill_y, label=label, **fill_kwargs)
        if legend and label:
            ax.legend()
    else:
        if contour_kwargs is None:
            contour_kwargs = {}
        contour_kwargs.setdefault("colors", "0.5")
        if contourf_kwargs is None:
            contourf_kwargs = {}
        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}

        # gridsize = (128, 128) if contour else (256, 256)

        # density, xmin, xmax, ymin, ymax = _fast_kde_2d(values, values2, gridsize=gridsize)
        g_s = complex(gridsize[0])
        x_x, y_y = np.mgrid[xmin:xmax:g_s, ymin:ymax:g_s]

        ax.grid(False)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if contour:
            qcfs = ax.contourf(x_x, y_y, density, antialiased=True, **contourf_kwargs)
            qcs = ax.contour(x_x, y_y, density, **contour_kwargs)
            if not fill_last:
                qcfs.collections[0].set_alpha(0)
                qcs.collections[0].set_alpha(0)
        else:
            ax.pcolormesh(x_x, y_y, density, **pcolormesh_kwargs)

    if backend_show(show):
        plt.show()

    return ax
