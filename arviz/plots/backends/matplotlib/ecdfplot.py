"""Matplotlib ecdfplot."""

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, create_axes_grid, backend_show


def plot_ecdf(
    x_coord,
    y_coord,
    x_bands,
    lower,
    higher,
    plot_kwargs,
    fill_kwargs,
    plot_outline_kwargs,
    figsize,
    fill_band,
    ax,
    show,
    backend_kwargs,
):
    """Matplotlib ecdfplot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    (figsize, _, _, _, _, _) = _scale_fig_size(figsize, None)
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True

    if ax is None:
        _, ax = create_axes_grid(1, backend_kwargs=backend_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}

    plot_kwargs.setdefault("where", "post")

    if fill_band:
        if fill_kwargs is None:
            fill_kwargs = {}
        fill_kwargs.setdefault("step", "post")
        fill_kwargs.setdefault("color", to_hex("C0"))
        fill_kwargs.setdefault("alpha", 0.2)
    else:
        if plot_outline_kwargs is None:
            plot_outline_kwargs = {}
        plot_outline_kwargs.setdefault("where", "post")
        plot_outline_kwargs.setdefault("color", to_hex("C0"))
        plot_outline_kwargs.setdefault("alpha", 0.2)

    ax.step(x_coord, y_coord, **plot_kwargs)

    if x_bands is not None:
        if fill_band:
            ax.fill_between(x_bands, lower, higher, **fill_kwargs)
        else:
            ax.plot(x_bands, lower, x_bands, higher, **plot_outline_kwargs)

    if backend_show(show):
        plt.show()

    return ax
