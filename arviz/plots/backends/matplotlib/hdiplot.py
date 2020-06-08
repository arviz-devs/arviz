"""Matplotlib hdiplot."""
import warnings
import matplotlib.pyplot as plt

from . import backend_show


def plot_hdi(ax, x_data, y_data, plot_kwargs, fill_kwargs, backend_kwargs, show):
    """Matplotlib hdi plot."""
    if backend_kwargs is not None:
        warnings.warn(
            (
                "Argument backend_kwargs has not effect in matplotlib.plot_hdi"
                "Supplied value won't be used"
            )
        )
    if ax is None:
        ax = plt.gca()
    ax.plot(x_data, y_data, **plot_kwargs)
    ax.fill_between(x_data, y_data[:, 0], y_data[:, 1], **fill_kwargs)

    if backend_show(show):
        plt.show()

    return ax
