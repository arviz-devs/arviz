"""Matplotlib hdiplot."""
import matplotlib.pyplot as plt

from . import backend_kwarg_defaults, backend_show


def plot_hdi(ax, x_data, y_data, plot_kwargs, fill_kwargs, backend_kwargs, show):
    """Matplotlib HDI plot."""
    if backend_kwargs is None:
        backend_kwargs = {}

    backend_kwargs = {
        **backend_kwarg_defaults(),
        **backend_kwargs,
    }
    if ax is None:
        _, ax = plt.subplots(1, 1, **backend_kwargs)

    ax.plot(x_data, y_data, **plot_kwargs)
    ax.fill_between(x_data, y_data[:, 0], y_data[:, 1], **fill_kwargs)

    if backend_show(show):
        plt.show()

    return ax
