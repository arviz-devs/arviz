"""Matplotlib hpdplot."""
import warnings
from matplotlib.pyplot import gca


def plot_hpd(ax, x_data, y_data, plot_kwargs, fill_kwargs, backend_kwargs):
    """Matplotlib hpd plot."""
    if backend_kwargs is not None:
        warnings.warn(
            (
                "Argument backend_kwargs has not effect in matplotlib.plot_hpd"
                "Supplied value won't be used"
            )
        )
    if ax is None:
        ax = gca()
    ax.plot(x_data, y_data, **plot_kwargs)
    ax.fill_between(x_data, y_data[:, 0], y_data[:, 1], **fill_kwargs)
    return ax
