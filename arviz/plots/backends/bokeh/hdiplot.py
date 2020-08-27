"""Bokeh hdiplot."""
import numpy as np

from ...plot_utils import _scale_fig_size, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid


def plot_hdi(ax, x_data, y_data, color, figsize, plot_kwargs, fill_kwargs, backend_kwargs, show):
    """Bokeh HDI plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    plot_kwargs["color"] = vectorized_to_hex(plot_kwargs.get("color", color))
    plot_kwargs.setdefault("alpha", 0)

    fill_kwargs = {} if fill_kwargs is None else fill_kwargs
    fill_kwargs["color"] = vectorized_to_hex(fill_kwargs.get("color", color))
    fill_kwargs.setdefault("alpha", 0.5)

    figsize, *_ = _scale_fig_size(figsize, None)

    if ax is None:
        ax = create_axes_grid(
            1,
            figsize=figsize,
            squeeze=True,
            backend_kwargs=backend_kwargs,
        )

    plot_kwargs.setdefault("line_color", plot_kwargs.pop("color"))
    plot_kwargs.setdefault("line_alpha", plot_kwargs.pop("alpha", 0))

    fill_kwargs.setdefault("fill_color", fill_kwargs.pop("color"))
    fill_kwargs.setdefault("fill_alpha", fill_kwargs.pop("alpha", 0))

    ax.patch(
        np.concatenate((x_data, x_data[::-1])),
        np.concatenate((y_data[:, 0], y_data[:, 1][::-1])),
        **fill_kwargs,
        **plot_kwargs
    )

    show_layout(ax, show)

    return ax
