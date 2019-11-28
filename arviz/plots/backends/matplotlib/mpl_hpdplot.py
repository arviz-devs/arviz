"""Matplotlib hpdplot."""
from matplotlib.pyplot import gca


def _plot_hpdplot(ax, x_data, y_data, plot_kwargs, fill_kwargs):
    if ax is None:
        ax = gca()
    ax.plot(x_data, y_data, **plot_kwargs)
    ax.fill_between(x_data, y_data[:, 0], y_data[:, 1], **fill_kwargs)
    return ax
