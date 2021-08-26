# pylint: disable=c-extension-no-member
"""Bokeh KDE Plot."""
from collections.abc import Callable
from numbers import Integral

from matplotlib import _contour
import numpy as np
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.glyphs import Scatter
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex
from matplotlib.pyplot import rcParams as mpl_rcParams

from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


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
    label,  # pylint: disable=unused-argument
    quantiles,
    rotated,
    contour,
    fill_last,
    figsize,
    textsize,  # pylint: disable=unused-argument
    plot_kwargs,
    fill_kwargs,
    rug_kwargs,
    contour_kwargs,
    contourf_kwargs,
    pcolormesh_kwargs,
    is_circular,  # pylint: disable=unused-argument
    ax,
    legend,  # pylint: disable=unused-argument
    backend_kwargs,
    show,
    return_glyph,
):
    """Bokeh kde plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    figsize, *_ = _scale_fig_size(figsize, textsize)

    if ax is None:
        ax = create_axes_grid(
            1,
            figsize=figsize,
            squeeze=True,
            backend_kwargs=backend_kwargs,
        )

    glyphs = []
    if values2 is None:
        if plot_kwargs is None:
            plot_kwargs = {}
        plot_kwargs.setdefault("line_color", mpl_rcParams["axes.prop_cycle"].by_key()["color"][0])

        if fill_kwargs is None:
            fill_kwargs = {}

        fill_kwargs.setdefault("fill_color", mpl_rcParams["axes.prop_cycle"].by_key()["color"][0])

        if rug:
            if rug_kwargs is None:
                rug_kwargs = {}

            rug_kwargs = rug_kwargs.copy()
            if "cds" in rug_kwargs:
                cds_rug = rug_kwargs.pop("cds")
                rug_varname = rug_kwargs.pop("y", "y")
            else:
                rug_varname = "y"
                cds_rug = ColumnDataSource({rug_varname: np.asarray(values)})

            rug_kwargs.setdefault("size", 8)
            rug_kwargs.setdefault("line_color", plot_kwargs["line_color"])
            rug_kwargs.setdefault("line_width", 1)
            rug_kwargs.setdefault("line_alpha", 0.35)
            if not rotated:
                rug_kwargs.setdefault("angle", np.pi / 2)
            if isinstance(cds_rug, dict):
                for _cds_rug in cds_rug.values():
                    if not rotated:
                        glyph = Scatter(x=rug_varname, y=0.0, marker="dash", **rug_kwargs)
                    else:
                        glyph = Scatter(x=0.0, y=rug_varname, marker="dash", **rug_kwargs)
                    ax.add_glyph(_cds_rug, glyph)
            else:
                if not rotated:
                    glyph = Scatter(x=rug_varname, y=0.0, marker="dash", **rug_kwargs)
                else:
                    glyph = Scatter(x=0.0, y=rug_varname, marker="dash", **rug_kwargs)
                ax.add_glyph(cds_rug, glyph)
            glyphs.append(glyph)

        x = np.linspace(lower, upper, len(density))

        if quantiles is not None:
            fill_kwargs.setdefault("fill_alpha", 0.75)
            fill_kwargs.setdefault("line_color", None)

            quantiles = sorted(np.clip(quantiles, 0, 1))
            if quantiles[0] != 0:
                quantiles = [0] + quantiles
            if quantiles[-1] != 1:
                quantiles = quantiles + [1]

            for quant_0, quant_1 in zip(quantiles[:-1], quantiles[1:]):
                idx = (density_q > quant_0) & (density_q < quant_1)
                if idx.sum():
                    patch_x = np.concatenate((x[idx], [x[idx][-1]], x[idx][::-1], [x[idx][0]]))
                    patch_y = np.concatenate(
                        (np.zeros_like(density[idx]), [density[idx][-1]], density[idx][::-1], [0])
                    )
                    if not rotated:
                        patch = ax.patch(patch_x, patch_y, **fill_kwargs)
                    else:
                        patch = ax.patch(patch_y, patch_x, **fill_kwargs)
                    glyphs.append(patch)
        else:
            if fill_kwargs.get("fill_alpha", False):
                patch_x = np.concatenate((x, [x[-1]], x[::-1], [x[0]]))
                patch_y = np.concatenate(
                    (np.zeros_like(density), [density[-1]], density[::-1], [0])
                )
                if not rotated:
                    patch = ax.patch(patch_x, patch_y, **fill_kwargs)
                else:
                    patch = ax.patch(patch_y, patch_x, **fill_kwargs)
                glyphs.append(patch)

            if label is not None:
                plot_kwargs.setdefault("legend_label", label)
            if not rotated:
                line = ax.line(x, density, **plot_kwargs)
            else:
                line = ax.line(density, x, **plot_kwargs)
            glyphs.append(line)

    else:
        if contour_kwargs is None:
            contour_kwargs = {}
        if contourf_kwargs is None:
            contourf_kwargs = {}
        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}

        g_s = complex(gridsize[0])
        x_x, y_y = np.mgrid[xmin:xmax:g_s, ymin:ymax:g_s]

        if contour:

            scaled_density, *scaled_density_args = _scale_axis(density)

            contour_generator = _contour.QuadContourGenerator(
                x_x, y_y, scaled_density, None, True, 0
            )

            levels = 9
            if "levels" in contourf_kwargs:
                levels = contourf_kwargs.pop("levels")
            if "levels" in contour_kwargs:
                levels = contour_kwargs.pop("levels")

            if isinstance(levels, Integral):
                levels_scaled = np.linspace(0, 1, levels + 2)
                levels = _rescale_axis(levels_scaled, scaled_density_args)
            else:
                levels_scaled_nonclip, *_ = _scale_axis(np.asarray(levels), scaled_density_args)
                levels_scaled = np.clip(levels_scaled_nonclip, 0, 1)

            cmap = contourf_kwargs.pop("cmap", "viridis")
            if isinstance(cmap, str):
                cmap = get_cmap(cmap)
            if isinstance(cmap, Callable):
                colors = [rgb2hex(item) for item in cmap(np.linspace(0, 1, len(levels_scaled) + 1))]
            else:
                colors = cmap

            contour_kwargs.update(contourf_kwargs)
            contour_kwargs.setdefault("line_alpha", 0.25)
            contour_kwargs.setdefault("fill_alpha", 1)

            for i, (level, level_upper, color) in enumerate(
                zip(levels_scaled[:-1], levels_scaled[1:], colors[1:])
            ):
                if not fill_last and (i == 0):
                    continue
                contour_kwargs_ = contour_kwargs.copy()
                contour_kwargs_.setdefault("line_color", color)
                contour_kwargs_.setdefault("fill_color", color)
                vertices, _ = contour_generator.create_filled_contour(level, level_upper)
                for seg in vertices:
                    # ax.multi_polygon would be better, but input is
                    # currently not suitable
                    # seg is 1 line that defines an area
                    # multi_polygon would need inner and outer edges
                    # as a line
                    patch = ax.patch(*seg.T, **contour_kwargs_)
                    glyphs.append(patch)

            if fill_last:
                ax.background_fill_color = colors[0]

            ax.xgrid.grid_line_color = None
            ax.ygrid.grid_line_color = None

            ax.x_range = Range1d(xmin, xmax)
            ax.y_range = Range1d(ymin, ymax)

        else:

            cmap = pcolormesh_kwargs.pop("cmap", "viridis")
            if isinstance(cmap, str):
                cmap = get_cmap(cmap)
            if isinstance(cmap, Callable):
                colors = [rgb2hex(item) for item in cmap(np.linspace(0, 1, 256))]
            else:
                colors = cmap

            image = ax.image(
                image=[density.T],
                x=xmin,
                y=ymin,
                dw=(xmax - xmin) / density.shape[0],
                dh=(ymax - ymin) / density.shape[1],
                palette=colors,
                **pcolormesh_kwargs
            )
            glyphs.append(image)
            ax.x_range.range_padding = ax.y_range.range_padding = 0

    show_layout(ax, show)

    if return_glyph:
        return ax, glyphs

    return ax


def _scale_axis(arr, args=None):
    if args:
        amin, amax = args
    else:
        amin, amax = arr.min(), arr.max()
    scaled_arr = arr - amin
    scaled_arr /= amax - amin
    return scaled_arr, amin, amax


def _rescale_axis(arr, args):
    amin, amax = args
    rescaled_arr = arr * (amax - amin)
    rescaled_arr += amin
    return rescaled_arr
