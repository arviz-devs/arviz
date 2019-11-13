"""Bokeh KDE Plot."""
import bokeh.plotting as bkp
from bokeh.models import ColumnDataSource, Dash
from bokeh.palettes import Viridis11
import matplotlib.pyplot as plt
import numpy as np
import warnings


def _plot_kde_bokeh(
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
    values2=None,
    rug=False,
    label=None,
    bw=4.5,
    quantiles=None,
    rotated=False,
    contour=True,
    fill_last=True,
    textsize=None,
    plot_kwargs=None,
    fill_kwargs=None,
    rug_kwargs=None,
    contour_kwargs=None,
    contourf_kwargs=None,
    pcolormesh_kwargs=None,
    ax=None,
    legend=True,
    show=True,
):
    if ax is None:
        ax = bkp.figure(width=500, height=500, output_backend="webgl")

    if legend and label is not None:
        plot_kwargs["legend_label"] = label

    if values2 is None:
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.setdefault("line_color", plt.rcParams["axes.prop_cycle"].by_key()["color"][0])

        default_color = plot_kwargs.get("color")

        if fill_kwargs is None:
            fill_kwargs = {}

        fill_kwargs.setdefault("color", default_color)

        if rug:
            if rug_kwargs is None:
                rug_kwargs = {}

            rug_kwargs = rug_kwargs.copy()
            if "cds" in rug_kwargs:
                cds_rug = rug_kwargs.pop("cds")
                rug_varname = rug_kwargs.pop("y", "y")
            else:
                rug_varname = "y"
                cds_rug = ColumnDataSource({rug_varname: values})

            rug_kwargs.setdefault("size", 8)
            rug_kwargs.setdefault("line_color", plot_kwargs["line_color"])
            rug_kwargs.setdefault("line_width", 1)
            rug_kwargs.setdefault("line_alpha", 0.35)
            rug_kwargs.setdefault("angle", np.pi / 2)
            if isinstance(cds_rug, dict):
                for _cds_rug in cds_rug.values():
                    glyph = Dash(x=rug_varname, y=0.0, **rug_kwargs)
                    ax.add_glyph(_cds_rug, glyph)
            else:
                glyph = Dash(x=rug_varname, y=0.0, **rug_kwargs)
                ax.add_glyph(cds_rug, glyph)

        x = np.linspace(lower, upper, len(density))
        ax.line(x, density, **plot_kwargs)
    else:
        if contour_kwargs is None:
            contour_kwargs = {}
        contour_kwargs.setdefault("colors", "0.5")
        if contourf_kwargs is None:
            contourf_kwargs = {}
        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}

        if contour:
            warnings.warn(
                "contour and contourf are not yet impelented on bokeh backend."
                "Use matplotlib interface to get these objects. Falls back to mpl",
                UserWarning,
            )
        ax.image(
            image=[density.T],
            x=xmin,
            y=ymin,
            dw=(xmax - xmin) / density.shape[0],
            dh=(ymax - ymin) / density.shape[1],
            palette=Viridis11,
            **pcolormesh_kwargs
        )
        ax.x_range.range_padding = ax.y_range.range_padding = 0

    if show:
        bkp.show(ax)
    return ax
