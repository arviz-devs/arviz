"""Matplotlib kdeplot."""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import _pylab_helpers
import matplotlib.ticker as mticker


from ...plot_utils import _scale_fig_size, _init_kwargs_dict
from . import backend_kwarg_defaults, backend_show, create_axes_grid, matplotlib_kwarg_dealiaser


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
    figsize,
    textsize,
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
    contour_kwargs,
    contourf_kwargs,
    pcolormesh_kwargs,
    is_circular,
    ax,
    legend,
    backend_kwargs,
    show,
    return_glyph,  # pylint: disable=unused-argument
):
    """Matplotlib kde plot."""
    backend_kwargs = _init_kwargs_dict(backend_kwargs)

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, *_, xt_labelsize, linewidth, markersize = _scale_fig_size(figsize, textsize)

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

    if values2 is None:
        plot_kwargs = matplotlib_kwarg_dealiaser(plot_kwargs, "plot")
        plot_kwargs.setdefault("color", "C0")

        default_color = plot_kwargs.get("color")

        fill_kwargs = matplotlib_kwarg_dealiaser(fill_kwargs, "hexbin")
        fill_kwargs.setdefault("color", default_color)

        rug_kwargs = matplotlib_kwarg_dealiaser(rug_kwargs, "plot")
        rug_kwargs.setdefault("marker", "_" if rotated else "|")
        rug_kwargs.setdefault("linestyle", "None")
        rug_kwargs.setdefault("color", default_color)
        rug_kwargs.setdefault("space", 0.2)

        plot_kwargs.setdefault("linewidth", linewidth)
        rug_kwargs.setdefault("markersize", 2 * markersize)

        rug_space = max(density) * rug_kwargs.pop("space")

        if is_circular:

            if is_circular == "radians":
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

                ticks_loc = ax.get_xticks()
                ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                ax.set_xticklabels(labels)

            x = np.linspace(-np.pi, np.pi, len(density))
            ax.set_yticklabels([])

        else:
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
                **fill_kwargs,
            )
        else:
            fill_kwargs.setdefault("alpha", 0)
            if fill_kwargs.get("alpha") == 0:
                label = plot_kwargs.setdefault("label", label)
                ax.plot(x, density, **plot_kwargs)
                fill_func(fill_x, fill_y, **fill_kwargs)
            else:
                label = fill_kwargs.setdefault("label", label)
                ax.plot(x, density, **plot_kwargs)
                fill_func(fill_x, fill_y, **fill_kwargs)
        if legend and label:
            ax.legend()
    else:
        contour_kwargs = matplotlib_kwarg_dealiaser(contour_kwargs, "contour")
        contour_kwargs.setdefault("colors", "0.5")
        contourf_kwargs = matplotlib_kwarg_dealiaser(contourf_kwargs, "contour")
        pcolormesh_kwargs = matplotlib_kwarg_dealiaser(pcolormesh_kwargs, "pcolormesh")
        pcolormesh_kwargs.setdefault("shading", "auto")

        g_s = complex(gridsize[0])
        x_x, y_y = np.mgrid[xmin:xmax:g_s, ymin:ymax:g_s]

        ax.grid(False)
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
