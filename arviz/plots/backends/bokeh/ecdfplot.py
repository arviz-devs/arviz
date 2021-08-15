"""Bokeh ecdfplot."""
from matplotlib.colors import to_hex

from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


def plot_ecdf(
    x_coord,
    y_coord,
    x_bands,
    lower,
    higher,
    confidence_bands,
    plot_kwargs,
    fill_kwargs,
    plot_outline_kwargs,
    figsize,
    fill_band,
    ax,
    show,
    backend_kwargs,
):
    """Bokeh ecdfplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    (figsize, *_) = _scale_fig_size(figsize, None)

    if ax is None:
        ax = create_axes_grid(
            1,
            figsize=figsize,
            squeeze=True,
            backend_kwargs=backend_kwargs,
        )

    if plot_kwargs is None:
        plot_kwargs = {}

    plot_kwargs.setdefault("mode", "after")

    if fill_band:
        if fill_kwargs is None:
            fill_kwargs = {}
        fill_kwargs.setdefault("fill_color", to_hex("C0"))
        fill_kwargs.setdefault("fill_alpha", 0.2)
    else:
        if plot_outline_kwargs is None:
            plot_outline_kwargs = {}
        plot_outline_kwargs.setdefault("color", to_hex("C0"))
        plot_outline_kwargs.setdefault("alpha", 0.2)

    if confidence_bands:
        ax.step(x_coord, y_coord, **plot_kwargs)

        if fill_band:
            ax.varea(x_bands, lower, higher, **fill_kwargs)
        else:
            ax.line(x_bands, lower, **plot_outline_kwargs)
            ax.line(x_bands, higher, **plot_outline_kwargs)
    else:
        ax.step(x_coord, y_coord, **plot_kwargs)

    show_layout(ax, show)

    return ax
