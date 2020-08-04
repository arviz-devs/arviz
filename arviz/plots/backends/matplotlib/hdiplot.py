"""Matplotlib hdiplot."""
import matplotlib.pyplot as plt

from . import backend_kwarg_defaults, backend_show
from ...plot_utils import matplotlib_kwarg_dealiaser, vectorized_to_hex


def plot_hdi(ax, x_data, y_data, color, plot_kwargs, fill_kwargs, backend_kwargs, show):
    """Matplotlib HDI plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }

    plot_kwargs = matplotlib_kwarg_dealiaser(plot_kwargs, "plot")
    plot_kwargs["color"] = vectorized_to_hex(plot_kwargs.get("color", color))
    plot_kwargs.setdefault("alpha", 0)

    fill_kwargs = matplotlib_kwarg_dealiaser(fill_kwargs, "fill_between")
    fill_kwargs["color"] = vectorized_to_hex(fill_kwargs.get("color", color))
    fill_kwargs.setdefault("alpha", 0.5)

    if ax is None:
        _, ax = plt.subplots(1, 1, **backend_kwargs)

    ax.plot(x_data, y_data, **plot_kwargs)
    ax.fill_between(x_data, y_data[:, 0], y_data[:, 1], **fill_kwargs)

    if backend_show(show):
        plt.show()

    return ax
