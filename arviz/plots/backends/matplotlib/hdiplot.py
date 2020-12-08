"""Matplotlib hdiplot."""
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers

from ...plot_utils import _scale_fig_size, vectorized_to_hex
from . import backend_kwarg_defaults, backend_show, create_axes_grid, matplotlib_kwarg_dealiaser


def plot_hdi(ax, x_data, y_data, color, figsize, plot_kwargs, fill_kwargs, backend_kwargs, show):
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

    figsize, *_ = _scale_fig_size(figsize, None)
    backend_kwargs.setdefault("figsize", figsize)
    backend_kwargs["squeeze"] = True

    if ax is None:
        fig_manager = _pylab_helpers.Gcf.get_active()
        if fig_manager is not None:
            ax = fig_manager.canvas.figure.gca()
        else:
            _, ax = create_axes_grid(
                1,
                backend_kwargs=backend_kwargs,
            )

    ax.plot(x_data, y_data, **plot_kwargs)
    ax.fill_between(x_data, y_data[:, 0], y_data[:, 1], **fill_kwargs)

    if backend_show(show):
        plt.show()

    return ax
