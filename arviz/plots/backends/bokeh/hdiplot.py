"""Bokeh hdiplot."""
import bokeh.plotting as bkp
import numpy as np

from . import backend_kwarg_defaults
from .. import show_layout
from ...plot_utils import vectorized_to_hex


def plot_hdi(ax, x_data, y_data, color, plot_kwargs, fill_kwargs, backend_kwargs, show):
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

    if ax is None:
        ax = bkp.figure(**backend_kwargs)

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
